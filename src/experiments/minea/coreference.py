"""
Coreference-split needle generation and paired injection for MINEA.

Generates two renderings of the same target triple in a single LLM call:
- ``flat_sentence``: a self-contained S-V-O sentence (current MINEA baseline).
- ``split_sentence``: 2-3 sentences where the canonical subject is introduced
  in one sentence, and the relation is expressed in a later sentence using
  only a referring expression (pronoun, definite description, abbreviation,
  citation).

Single-pass paired generation guarantees the (S, P, O) target is identical
across both renderings — there is no separate transformation step that could
fail or silently drift the fact.

Paired injection (``inject_paired_needles``) chooses how many needles to
inject based on the LONGER (split) rendering, then injects the same needle
indices at the same positions in both copies. This eliminates the confound
where flat and split would otherwise have different ``n_injected`` counts.
"""

import asyncio
import json
import logging
import re
from typing import Any

import dspy

from .needles import inject_needles

logger = logging.getLogger(__name__)


class _GeneratePairedNeedles(dspy.Signature):
    """
    You are a research paper analyst. Given an excerpt from an academic paper,
    generate synthetic factual triples (subject-predicate-object) that:

    1. Are thematically consistent with the paper's domain and content
    2. Could plausibly appear in a paper like this, but do NOT exist in the excerpt
    3. Are specific and concrete (not generic or vague)
    4. Use clear, well-defined entities and relationships

    For EACH triple, produce TWO natural-language renderings:

    A) ``flat_sentence`` — a single self-contained sentence containing the
       subject, predicate, and object verbatim.

       Example for triple (ResNet-50, achieved accuracy on, 97.3% on MIT-BIH):
         "ResNet-50 achieved 97.3% classification accuracy on the
          MIT-BIH arrhythmia dataset."

    B) ``split_sentence`` — TWO OR THREE connected sentences that distribute
       the same fact across sentence boundaries via referring expressions.

       Constraints (ALL must hold):
       - The canonical subject is introduced in the first sentence in a
         context that does NOT contain the predicate or object.
       - The fact (predicate + object) is expressed in a LATER sentence
         using ONLY a referring expression for the subject — pronoun
         ("it", "they"), definite description ("the model", "the
         architecture", "the method"), abbreviation, or citation-style
         reference ("the approach proposed by He et al.").
       - The full canonical subject string MUST NOT appear in the
         fact-bearing sentence.
       - Only ONE plausible antecedent exists for the referring expression.
       - The original target fact (subject + predicate + object) must be
         preserved exactly — do not paraphrase the fact, only its delivery.
       - The prose should resemble real academic writing, not a synthetic
         puzzle.

       Example for the same triple:
         "He et al. introduced ResNet-50 for image classification in 2015.
          The architecture was later adapted for cardiac signal analysis,
          where it achieved 97.3% accuracy on the MIT-BIH arrhythmia
          dataset."

    Generate exactly N distinct needle triples. Each should be independently
    verifiable (avoid triples that only make sense together).

    Return a JSON array of objects with fields:
    - subject: the subject entity (canonical form)
    - predicate: the relationship/action
    - object: the object entity
    - keywords: list of 3-5 distinctive keywords from S/P/O for matching
    - flat_sentence: the flat single-sentence rendering
    - split_sentence: the multi-sentence coreference-split rendering
    """

    paper_excerpt: str = dspy.InputField(
        desc="Excerpt from an academic paper (context for domain relevance)"
    )
    n_needles: int = dspy.InputField(desc="Number of needle triples to generate")
    needles_json: str = dspy.OutputField(
        desc=(
            "JSON array of needle objects with subject, predicate, object, "
            "keywords, flat_sentence, split_sentence"
        )
    )


def _strip_code_fence(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\n?", "", raw)
        raw = re.sub(r"\n?```\s*$", "", raw)
        raw = raw.strip()
    return raw


async def generate_paired_needles(
    paper_text: str,
    n_needles: int,
    model: str,
    initial_delay: float,
    max_attempts: int,
) -> list[dict[str, Any]]:
    """Generate paired (flat, split) needle triples for a paper excerpt.

    Returns needles with the shape::

        {
            "subject": str,
            "predicate": str,
            "object": str,
            "keywords": list[str],
            "flat_sentence": str,
            "split_sentence": str,
        }

    Needles missing either rendering or any of the S/P/O fields are dropped.
    """
    lm = dspy.LM(model, max_tokens=4096)
    predict = dspy.Predict(_GeneratePairedNeedles)

    excerpt = paper_text[:5000]

    delay = initial_delay
    for attempt in range(1, max_attempts + 1):
        try:
            with dspy.context(lm=lm):
                pred = await predict.acall(paper_excerpt=excerpt, n_needles=n_needles)

            raw = _strip_code_fence(getattr(pred, "needles_json", "[]") or "[]")
            data = json.loads(raw)
            if not isinstance(data, list):
                raise ValueError("Response is not a JSON array")

            needles: list[dict[str, Any]] = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                s = str(item.get("subject", "")).strip()
                p = str(item.get("predicate", "")).strip()
                o = str(item.get("object", "")).strip()
                flat = str(item.get("flat_sentence", "")).strip()
                split = str(item.get("split_sentence", "")).strip()
                if not (s and p and o and flat and split):
                    continue

                kws_raw = item.get("keywords", [])
                if not isinstance(kws_raw, list):
                    kws_raw = []
                validated_kws = [str(k).strip() for k in kws_raw if str(k).strip()]

                needles.append({
                    "subject": s,
                    "predicate": p,
                    "object": o,
                    "keywords": validated_kws,
                    "flat_sentence": flat,
                    "split_sentence": split,
                })

            if len(needles) < n_needles:
                logger.warning(
                    f"Only generated {len(needles)}/{n_needles} valid paired needles"
                )

            return needles

        except Exception as e:
            if attempt < max_attempts:
                logger.warning(
                    f"[paired-needle-gen] attempt {attempt} failed: {e}, retrying in {delay}s"
                )
                await asyncio.sleep(delay)
                delay *= 2
            else:
                logger.error(f"[paired-needle-gen] all attempts failed: {e}")
                return []

    return []


def _project_needle(paired: dict[str, Any], rendering: str) -> dict[str, Any]:
    """Project a paired needle to the {sentence: ...} shape the rest of the
    pipeline (inject_needles, evaluate_needle_capture) expects."""
    sentence_key = "flat_sentence" if rendering == "flat" else "split_sentence"
    return {
        "subject": paired["subject"],
        "predicate": paired["predicate"],
        "object": paired["object"],
        "keywords": paired["keywords"],
        "sentence": paired[sentence_key],
    }


def inject_paired_needles(
    chunk_text: str,
    paired_needles: list[dict[str, Any]],
    min_frac: float,
    max_frac: float,
    niah_epsilon: float,
    seed: int,
) -> tuple[str, str, float, float, int]:
    """Inject the same needle indices at the same positions in two renderings.

    The number of needles to inject is chosen from the LONGER (split)
    rendering so the split copy stays within the 10-30% NIAH band. The
    flat copy uses the same count, which means the flat copy may sit
    below 10% — that asymmetry is documented in metadata as a known
    limitation; we do not claim a directional bias on the resulting gap.

    Returns:
        (enriched_flat, enriched_split, frac_flat, frac_split, n_injected)
    """
    if not paired_needles:
        return chunk_text, chunk_text, 0.0, 0.0, 0

    # Decide n_to_inject from the split (longer) rendering. We do this by
    # asking inject_needles itself, on a throwaway copy, what it would
    # choose for the split rendering — that keeps the budgeting math in
    # one place rather than re-implementing it here.
    split_needles_for_budget = [_project_needle(p, "split") for p in paired_needles]
    _, _, n_to_inject = inject_needles(
        chunk_text=chunk_text,
        needles=split_needles_for_budget,
        min_frac=min_frac,
        max_frac=max_frac,
        niah_epsilon=niah_epsilon,
        seed=seed,
    )

    # Now do the real two injections, each forcing the same n_to_inject
    # and using the same seed → same selected indices, same positions.
    flat_needles = [_project_needle(p, "flat") for p in paired_needles]
    split_needles = [_project_needle(p, "split") for p in paired_needles]

    enriched_flat, frac_flat, n_flat = inject_needles(
        chunk_text=chunk_text,
        needles=flat_needles,
        min_frac=min_frac,
        max_frac=max_frac,
        niah_epsilon=niah_epsilon,
        seed=seed,
        force_n_to_inject=n_to_inject,
    )
    enriched_split, frac_split, n_split = inject_needles(
        chunk_text=chunk_text,
        needles=split_needles,
        min_frac=min_frac,
        max_frac=max_frac,
        niah_epsilon=niah_epsilon,
        seed=seed,
        force_n_to_inject=n_to_inject,
    )

    # Hard-fail invariant: paired injection MUST produce identical counts.
    # If this assertion ever fires, the flat-vs-split comparison is
    # contaminated and the resulting numbers are not defensible.
    if n_flat != n_split or n_flat != n_to_inject:
        raise RuntimeError(
            f"Paired injection invariant violated: "
            f"requested={n_to_inject}, flat={n_flat}, split={n_split}. "
            "Both renderings must inject the same needle count."
        )

    logger.info(
        f"Paired injection: n_injected={n_flat}, "
        f"frac_flat={frac_flat:.1%}, frac_split={frac_split:.1%}"
    )

    return enriched_flat, enriched_split, frac_flat, frac_split, n_flat
