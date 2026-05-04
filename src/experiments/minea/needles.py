"""
Needle generation and injection for MINEA evaluation.

Handles:
- Synthetic needle triple generation
- Needle injection with NIAH compliance (10-30% fraction)
- Random placement within document chunks
"""

import asyncio
from collections import defaultdict
import json
import logging
import random
import re
from typing import Any

import dspy

logger = logging.getLogger(__name__)


class _GenerateNeedleTriples(dspy.Signature):
    """
    You are a research paper analyst. Given an excerpt from an academic paper,
    generate synthetic factual triples (subject-predicate-object) that:

    1. Are thematically consistent with the paper's domain and content
    2. Could plausibly appear in a paper like this, but do NOT exist in the excerpt
    3. Are specific and concrete (not generic or vague)
    4. Use clear, well-defined entities and relationships

    Generate exactly N distinct needle triples. Each should be independently
    verifiable (avoid triples that only make sense together).

    Return a JSON array of objects with:
    - subject: the subject entity
    - predicate: the relationship/action
    - object: the object entity
    - keywords: list of 3-5 distinctive keywords from S/P/O for matching
    - sentence: natural language sentence expressing this triple
    """

    paper_excerpt: str = dspy.InputField(
        desc="Excerpt from an academic paper (context for domain relevance)"
    )
    n_needles: int = dspy.InputField(desc="Number of needle triples to generate")
    needles_json: str = dspy.OutputField(
        desc="JSON array of needle objects with subject, predicate, object, keywords, sentence"
    )


async def generate_needles(
    paper_text: str,
    n_needles: int,
    needle_gen_model: str,
    initial_delay: float,
    max_attempts: int,
) -> list[dict[str, Any]]:
    """Generate needle triples for a given paper excerpt."""
    lm = dspy.LM(needle_gen_model, max_tokens=4096)
    predict = dspy.Predict(_GenerateNeedleTriples)

    # use first ~5000 chars as context for needle generation
    # (matches the chunk size used for injection to ensure contextual relevance)
    excerpt = paper_text[:5000]

    delay = initial_delay
    for attempt in range(1, max_attempts + 1):
        try:
            with dspy.context(lm=lm):
                pred = await predict.acall(paper_excerpt=excerpt, n_needles=n_needles)

            raw = getattr(pred, "needles_json", "[]").strip()
            # strip markdown code fence if present
            if raw.startswith("```"):
                raw = re.sub(r"^```\w*\n?", "", raw)
                raw = re.sub(r"\n?```\s*$", "", raw)
                raw = raw.strip()

            data = json.loads(raw)
            if not isinstance(data, list):
                raise ValueError("Response is not a JSON array")

            # validate structure
            needles = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                # Convert to strings and strip
                s = str(item.get("subject", "")).strip()
                p = str(item.get("predicate", "")).strip()
                o = str(item.get("object", "")).strip()
                sent = str(item.get("sentence", "")).strip()
                if not (s and p and o and sent):
                    continue

                # Validate keywords
                kws_raw = item.get("keywords", [])
                if not isinstance(kws_raw, list):
                    kws_raw = []
                validated_kws = [str(k).strip() for k in kws_raw if str(k).strip()]

                needles.append({
                    "subject": s,
                    "predicate": p,
                    "object": o,
                    "sentence": sent,
                    "keywords": validated_kws,
                })

            if len(needles) < n_needles:
                logger.warning(
                    f"Only generated {len(needles)}/{n_needles} valid needles"
                )

            return needles

        except Exception as e:
            if attempt < max_attempts:
                logger.warning(
                    f"[needle-gen] attempt {attempt} failed: {e}, retrying in {delay}s"
                )
                await asyncio.sleep(delay)
                delay *= 2
            else:
                logger.error(f"[needle-gen] all attempts failed: {e}")
                return []


def inject_needles(
    chunk_text: str,
    needles: list[dict[str, Any]],
    min_frac: float,
    max_frac: float,
    niah_epsilon: float,
    seed: int,
) -> tuple[str, float, int]:
    """
    Inject needle sentences into chunk text at random positions.

    Per Seitl et al. (2024): "We scatter several needles at random over the
    text document body (such that the inserted needles fill 10 to 30% of
    the enriched text)."

    STRICT ENFORCEMENT: Injects needles to achieve 10-30% needle fraction.
    Adjusts number of needles injected to stay within bounds.

    Inserts needles between sentences to maintain natural flow.

    Args:
        chunk_text: Original document text
        needles: List of needle dictionaries (with 'sentence' key)
        min_frac: Minimum needle fraction (default 0.1 = 10%)
        max_frac: Maximum needle fraction (default 0.3 = 30%)
        niah_epsilon: Tolerance for boundary checks
        seed: Random seed for reproducibility

    Returns:
        (enriched_text, needle_fraction, n_injected): enriched text, actual fraction,
        and number of needles injected (may be < len(needles) for compliance)
    """
    # split into sentences
    sentences = re.split(r"(?<=[.!?])\s+", chunk_text.strip())
    sentences = [s for s in sentences if s.strip()]  # Remove empty strings
    if not sentences:
        logger.warning("No sentences found in chunk, treating entire text as one sentence")
        sentences = [chunk_text.strip()]

    n_available = len(needles)
    if n_available == 0:
        return chunk_text, 0.0, 0

    original_len = len(chunk_text)

    # Calculate needle lengths
    needle_lens = [len(n["sentence"]) for n in needles]
    total_needle_len = sum(needle_lens)
    avg_needle_len = total_needle_len / n_available

    # Calculate range of needles that would put us in 10-30% range
    # For min_frac: needle_len = original_len * min_frac / (1 - min_frac)
    # For max_frac: needle_len = original_len * max_frac / (1 - max_frac)
    min_needle_len = original_len * min_frac / (1 - min_frac)
    max_needle_len = original_len * max_frac / (1 - max_frac)

    # Find optimal number of needles
    # Prefer more needles (better coverage) but stay within max_frac
    min_n = max(1, int(min_needle_len / avg_needle_len))
    max_n = int(max_needle_len / avg_needle_len)

    # Clamp to available needles
    min_n = min(min_n, n_available)
    max_n = min(max_n, n_available)

    # Check if constraint is satisfiable
    if max_n < min_n:
        # Cannot satisfy 10-30% range with available needles
        # Inject all available and accept the constraint violation
        n_to_inject = n_available
        logger.warning(
            f"Cannot reach {min_frac:.0%} with {n_available} needles "
            f"(avg_len={avg_needle_len:.0f}) in {original_len}-char doc. "
            f"Will inject all {n_available} needles (may be <10%)."
        )
    else:
        # Prefer more needles (better coverage) while staying within max_frac
        n_to_inject = max_n

    # Calculate actual length with selected needles
    actual_needle_len = sum(needle_lens[:n_to_inject])

    # Verify this will be in range
    enriched_len_projected = original_len + actual_needle_len
    projected_frac = actual_needle_len / enriched_len_projected if enriched_len_projected > 0 else 0.0

    logger.info(
        f"Needle injection: doc={original_len} chars, injecting {n_to_inject}/{n_available} needles, "
        f"projected fraction={projected_frac:.1%}"
    )

    # Select needles to inject (first n_to_inject)
    needles_to_inject = needles[:n_to_inject]

    # Select random positions for needle insertion
    # Use seeded RNG for reproducibility
    rng = random.Random(seed)

    # Map position -> list of needle indices to handle multiple needles per position
    position_to_needles = defaultdict(list)

    if len(sentences) >= n_to_inject:
        # One needle per position (random sampling without replacement)
        positions = sorted(rng.sample(range(len(sentences)), n_to_inject))
        for needle_idx, pos in enumerate(positions):
            position_to_needles[pos].append(needle_idx)
    elif len(sentences) == 1:
        # Edge case: single sentence, all needles cluster at position 0
        logger.warning(f"Only 1 sentence for {n_to_inject} needles - all will cluster at position 0")
        position_to_needles[0].extend(range(n_to_inject))
    else:
        # More needles than sentences: distribute randomly with replacement
        for needle_idx in range(n_to_inject):
            pos = rng.randint(0, len(sentences) - 1)
            position_to_needles[pos].append(needle_idx)

    # Insert needles at calculated positions
    enriched = []
    for i, sent in enumerate(sentences):
        enriched.append(sent)
        # Insert all needles mapped to this position
        if i in position_to_needles:
            for needle_idx in position_to_needles[i]:
                enriched.append(needles_to_inject[needle_idx]["sentence"])

    enriched_text = " ".join(enriched)

    # Calculate actual needle fraction
    enriched_len = len(enriched_text)
    needle_fraction = (enriched_len - original_len) / enriched_len if enriched_len > 0 else 0.0

    # Validate NIAH requirement (10-30%) - should always pass now
    if not (min_frac - niah_epsilon <= needle_fraction <= max_frac + niah_epsilon):
        logger.warning(
            f"⚠️ Needle fraction {needle_fraction:.1%} still outside NIAH range after adjustment! "
            f"Original={original_len}, enriched={enriched_len}, injected={n_to_inject} needles"
        )
    else:
        logger.info(f"✓ Needle fraction {needle_fraction:.1%} within NIAH 10-30% range")

    return enriched_text, needle_fraction, n_to_inject
