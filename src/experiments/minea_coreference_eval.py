"""
Experiment: MINEA with coreference-split needles.

Companion to minea_triple_eval.py. For each generated triple, we produce
TWO renderings — a flat S-V-O sentence (the current MINEA baseline) and a
multi-sentence "coreference-split" rendering where the relation is expressed
via a referring expression (pronoun, definite description, abbreviation,
citation) in a sentence other than the one that introduces the canonical
subject. We inject the same needle indices at the same positions in both
renderings, run extraction with each candidate model on each rendering,
and report side-by-side MINEA scores.

Single-pass paired generation guarantees the (S, P, O) target is identical
across renderings. Paired injection guarantees identical needle counts and
positions. This means the only thing varying between conditions is the
surface rendering of each needle — exactly the difficulty mechanism we want
to isolate.

Usage:
    cd /home/ubuntu/X-RAG/src
    PYTHONPATH=. python -m experiments.minea_coreference_eval

Output:
    src/experiments/minea_coreference_results.json
"""

import asyncio
import json
import logging
from statistics import mean

import dspy
from dspy.adapters.chat_adapter import ChatAdapter
from prettytable import PrettyTable

from xrag.paths import CACHE_DIR, DATASETS_DIR, SRC

# Re-import all configuration from the flat baseline runner so the two
# experiments stay in sync. The `if __name__ == "__main__"` guard in
# minea_triple_eval.py ensures importing it does not trigger a run.
from experiments.minea_triple_eval import (
    CANDIDATE_MODELS,
    NEEDLE_GEN_MODEL,
    JUDGE_MODEL,
    EMBED_MODEL,
    SAMPLE_SIZE,
    NEEDLES_PER_DOC,
    SEED,
    DATASET,
    INITIAL_DELAY,
    MAX_ATTEMPTS,
    MIN_CHUNK_TOKENS,
    EMBED_BATCH_LIMIT,
    KEYWORD_MATCH_THRESHOLD,
    SEMANTIC_SIM_THRESHOLD,
    ESTIMATED_NEEDLE_LENGTH,
    MAX_CONCURRENT_JUDGE_CALLS,
    MIN_CONTENT_LENGTH,
    MAX_AVG_LINE_LEN_TOC,
    CITATION_DENSITY_THRESHOLD,
    NIAH_EPSILON,
)

from experiments.minea.coreference import (
    generate_paired_needles,
    inject_paired_needles,
)
from experiments.minea.extraction import run_extraction_on_enriched_chunk
from experiments.minea.evaluation import evaluate_needle_capture
from experiments.minea.data_loader import load_sample_documents

# ── Configuration ────────────────────────────────────────────────────────────

# Use the same model for paired generation as the flat baseline uses for
# single-rendering generation. Override here if needed for ablations.
COREFERENCE_GEN_MODEL = NEEDLE_GEN_MODEL

OUTPUT_DIR = SRC / "experiments"
OUTPUT_FILE = OUTPUT_DIR / "minea_coreference_results.json"

LIMITATIONS = [
    "Quality validation of split renderings is deferred; the reported flat-vs-split gap is a lower bound on the true coreference effect.",
    "Split renderings are intrinsically longer than flat renderings; a residual length confound exists even with paired injection. The flat condition's needle fraction may sit below the 10% NIAH lower bound because n_to_inject is sized for the longer split rendering.",
    "exact/keyword metric drops should be interpreted with caution — they may reflect surface-form changes (pronouns) rather than capture failures.",
    "Semantic-match embedding uses the flat-form sentence for BOTH renderings (rather than the document-side rendering) so cosine-similarity comparisons aren't degraded by the surface-area mismatch between long split sentences and short extracted triples.",
]

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)


# ── Main Experiment ──────────────────────────────────────────────────────────


async def _evaluate_one(
    enriched_doc: dict,
    rendering: str,
    model_name: str,
    model_id: str,
) -> dict:
    """Run extraction + evaluation for a single (document, rendering) pair."""
    enriched_text = enriched_doc[f"enriched_{rendering}"]

    extraction = await run_extraction_on_enriched_chunk(
        model_name=model_name,
        model_id=model_id,
        original_text=enriched_doc["original_text"],
        enriched_text=enriched_text,
        initial_delay=INITIAL_DELAY,
        max_attempts=MAX_ATTEMPTS,
    )

    if "error" in extraction:
        logger.error(f"[{model_name}/{rendering}] Extraction failed: {extraction['error']}")
        return {
            "source_file": enriched_doc["source_file"],
            "extraction_failed": True,
            "error": extraction["error"],
        }

    needles_for_eval = enriched_doc[f"needles_{rendering}"]
    try:
        eval_result = await evaluate_needle_capture(
            needles=needles_for_eval,
            extraction_result=extraction,
            embed_model=EMBED_MODEL,
            judge_model=JUDGE_MODEL,
            embed_batch_limit=EMBED_BATCH_LIMIT,
            initial_delay=INITIAL_DELAY,
            max_attempts=MAX_ATTEMPTS,
            keyword_match_threshold=KEYWORD_MATCH_THRESHOLD,
            semantic_sim_threshold=SEMANTIC_SIM_THRESHOLD,
            max_concurrent_judge_calls=MAX_CONCURRENT_JUDGE_CALLS,
        )
    except Exception as e:
        logger.error(
            f"[{model_name}/{rendering}] Evaluation failed on "
            f"{enriched_doc['source_file']}: {e}"
        )
        return {
            "source_file": enriched_doc["source_file"],
            "evaluation_failed": True,
            "error": str(e),
        }

    logger.info(
        f"[{model_name}/{rendering}] {enriched_doc['source_file'][:40]}: "
        f"{extraction['triple_count']} triples, "
        f"MINEA_judge={eval_result['minea_judge']:.2f}, "
        f"MINEA_any={eval_result['minea_any']:.2f}"
    )

    return {
        "source_file": enriched_doc["source_file"],
        "extraction": extraction,
        "evaluation": eval_result,
    }


def _aggregate(model_results: list[dict]) -> dict:
    valid_evals = [
        r["evaluation"]
        for r in model_results
        if "evaluation" in r and r["evaluation"]["n_needles"] > 0
    ]
    if not valid_evals:
        return {"error": "No valid evaluations"}
    agg = {
        "n_documents": len(model_results),
        "n_valid_evals": len(valid_evals),
        "total_needles": sum(e["n_needles"] for e in valid_evals),
    }
    for metric in ("exact", "keyword", "semantic", "judge", "any"):
        agg[f"mean_minea_{metric}"] = round(
            mean(e[f"minea_{metric}"] for e in valid_evals), 4
        )
    return agg


async def run_experiment() -> None:
    dspy.configure(adapter=ChatAdapter())

    logger.info(f"Loading {SAMPLE_SIZE} documents from {DATASET} (seed={SEED})")
    documents = await load_sample_documents(
        datasets_dir=DATASETS_DIR,
        cache_dir=CACHE_DIR,
        dataset=DATASET,
        sample_size=SAMPLE_SIZE,
        min_chunk_tokens=MIN_CHUNK_TOKENS,
        seed=SEED,
        min_content_length=MIN_CONTENT_LENGTH,
        max_avg_line_len_toc=MAX_AVG_LINE_LEN_TOC,
        citation_density_threshold=CITATION_DENSITY_THRESHOLD,
    )
    if not documents:
        logger.error("No documents loaded — aborting")
        return

    # ── Generate paired needles ──
    logger.info(
        f"Generating paired (flat, split) needles for {len(documents)} documents"
    )
    doc_paired_needles = []
    for doc in documents:
        try:
            doc_len = len(doc["text"])
            # Use the same adaptive-count heuristic as the flat baseline.
            # Note: split sentences are 2-3x longer, so paired_inject's NIAH
            # budgeting will choose a smaller n_to_inject than the flat case
            # would. That's intentional — it keeps split inside the 10-30%
            # band, with the flat copy slightly under-budget on length.
            min_needle_len_total = doc_len * 0.1 / (1 - 0.1)
            min_needles_needed = max(
                NEEDLES_PER_DOC,
                int(min_needle_len_total / ESTIMATED_NEEDLE_LENGTH) + 1,
            )
            n_to_generate = min(min_needles_needed, NEEDLES_PER_DOC * 3)

            paired = await generate_paired_needles(
                paper_text=doc["text"],
                n_needles=n_to_generate,
                model=COREFERENCE_GEN_MODEL,
                initial_delay=INITIAL_DELAY,
                max_attempts=MAX_ATTEMPTS,
            )
            doc_paired_needles.append(paired)
            logger.info(
                f"Generated {len(paired)} paired needles for "
                f"{doc['source_file'][:40]} (doc_len={doc_len})"
            )
        except Exception as e:
            logger.error(
                f"Failed to generate paired needles for {doc['source_file']}: {e}"
            )
            doc_paired_needles.append([])

    # ── Paired injection ──
    # NOTE: documents whose paired-needle generation produced zero needles are
    # recorded in `n_docs_with_no_needles` for transparency, but excluded from
    # extraction (no point running the model on a non-enriched chunk). The
    # flat baseline runner does the same implicitly by aggregating only
    # evaluations with `n_needles > 0`, so the cross-experiment doc set
    # remains comparable.
    enriched_docs = []
    n_docs_with_no_needles = 0
    for doc, paired in zip(documents, doc_paired_needles):
        if not paired:
            logger.warning(
                f"Skipping {doc['source_file']}: no paired needles generated"
            )
            n_docs_with_no_needles += 1
            continue

        enriched_flat, enriched_split, frac_flat, frac_split, n_injected = (
            inject_paired_needles(
                chunk_text=doc["text"],
                paired_needles=paired,
                min_frac=0.1,
                max_frac=0.3,
                niah_epsilon=NIAH_EPSILON,
                seed=SEED,
            )
        )

        injected_paired = paired[:n_injected]

        # The evaluator's `sentence` field drives semantic-match embedding.
        # We deliberately use `flat_sentence` for BOTH conditions so the
        # needle-identity embedding stays the same length (~50 chars) in
        # both renderings. If we used `split_sentence` (~150 chars) for
        # the split condition, the cosine similarity vs. short extracted
        # triples would be systematically depressed for surface-area
        # reasons unrelated to coreference difficulty — a confound. The
        # rendering only differs on the *document* side (enriched_flat
        # vs enriched_split), which is what the extractor reads.
        enriched_docs.append({
            "source_file": doc["source_file"],
            "original_text": doc["text"],
            "enriched_flat": enriched_flat,
            "enriched_split": enriched_split,
            "needles_flat": [
                {**n, "sentence": n["flat_sentence"]} for n in injected_paired
            ],
            "needles_split": [
                {**n, "sentence": n["flat_sentence"]} for n in injected_paired
            ],
            "needle_fraction_flat": frac_flat,
            "needle_fraction_split": frac_split,
            "n_needles_generated": len(paired),
            "n_needles_injected": n_injected,
        })

    # ── Run extraction + evaluation per model, both renderings ──
    results: dict = {}
    for model_name, model_id in CANDIDATE_MODELS.items():
        logger.info(f"{'=' * 60}")
        logger.info(f"Running flat + split extraction with {model_name}")
        logger.info(f"{'=' * 60}")

        flat_per_doc: list[dict] = []
        split_per_doc: list[dict] = []
        for enriched_doc in enriched_docs:
            flat_per_doc.append(
                await _evaluate_one(enriched_doc, "flat", model_name, model_id)
            )
            split_per_doc.append(
                await _evaluate_one(enriched_doc, "split", model_name, model_id)
            )

        flat_agg = _aggregate(flat_per_doc)
        split_agg = _aggregate(split_per_doc)

        results[model_name] = {
            "model_id": model_id,
            "flat": flat_agg,
            "split": split_agg,
            "per_document_flat": flat_per_doc,
            "per_document_split": split_per_doc,
        }

        logger.info(
            f"[{model_name}] flat MINEA: "
            f"judge={flat_agg.get('mean_minea_judge', 0):.2%}, "
            f"any={flat_agg.get('mean_minea_any', 0):.2%}"
        )
        logger.info(
            f"[{model_name}] split MINEA: "
            f"judge={split_agg.get('mean_minea_judge', 0):.2%}, "
            f"any={split_agg.get('mean_minea_any', 0):.2%}"
        )

    # ── NIAH compliance summary ──
    flat_fracs = [d["needle_fraction_flat"] for d in enriched_docs]
    split_fracs = [d["needle_fraction_split"] for d in enriched_docs]

    def _frac_stats(fracs: list[float]) -> dict:
        if not fracs:
            return {"mean": 0.0, "min": 0.0, "max": 0.0, "compliant": False, "violations": 0}
        compliant = all(0.1 - NIAH_EPSILON <= f <= 0.3 + NIAH_EPSILON for f in fracs)
        violations = sum(
            1 for f in fracs
            if f < 0.1 - NIAH_EPSILON or f > 0.3 + NIAH_EPSILON
        )
        return {
            "mean": mean(fracs),
            "min": min(fracs),
            "max": max(fracs),
            "compliant": compliant,
            "violations": violations,
        }

    flat_stats = _frac_stats(flat_fracs)
    split_stats = _frac_stats(split_fracs)

    # ── Length statistics for confound analysis ──
    flat_lens: list[int] = []
    split_lens: list[int] = []
    for d in enriched_docs:
        flat_lens.extend(len(n["flat_sentence"]) for n in d["needles_flat"])
        split_lens.extend(len(n["split_sentence"]) for n in d["needles_split"])
    mean_flat_len = mean(flat_lens) if flat_lens else 0.0
    mean_split_len = mean(split_lens) if split_lens else 0.0
    length_ratio = (mean_split_len / mean_flat_len) if mean_flat_len > 0 else 0.0

    # ── Save results ──
    OUTPUT_DIR.mkdir(exist_ok=True)
    total_generated = sum(d["n_needles_generated"] for d in enriched_docs)
    total_injected = sum(d["n_needles_injected"] for d in enriched_docs)

    output = {
        "metadata": {
            "experiment": "coreference_split",
            "dataset": DATASET,
            "sample_size": SAMPLE_SIZE,
            "needles_per_doc": NEEDLES_PER_DOC,
            "needles_generated_total": total_generated,
            "needles_injected_total": total_injected,
            "n_docs_with_no_needles": n_docs_with_no_needles,
            "mean_flat_sentence_len_chars": round(mean_flat_len, 1),
            "mean_split_sentence_len_chars": round(mean_split_len, 1),
            "split_to_flat_length_ratio": round(length_ratio, 2),
            "seed": SEED,
            "needle_gen_model": NEEDLE_GEN_MODEL,
            "coreference_gen_model": COREFERENCE_GEN_MODEL,
            "judge_model": JUDGE_MODEL,
            "embed_model": EMBED_MODEL,
            "keyword_match_threshold": KEYWORD_MATCH_THRESHOLD,
            "semantic_sim_threshold": SEMANTIC_SIM_THRESHOLD,
            "needle_fraction_flat_mean": flat_stats["mean"],
            "needle_fraction_flat_range": [flat_stats["min"], flat_stats["max"]],
            "needle_fraction_split_mean": split_stats["mean"],
            "needle_fraction_split_range": [split_stats["min"], split_stats["max"]],
            "niah_compliance_strict_flat": flat_stats["compliant"],
            "niah_compliance_strict_split": split_stats["compliant"],
            "niah_violations_flat": flat_stats["violations"],
            "niah_violations_split": split_stats["violations"],
            "limitations": LIMITATIONS,
        },
        "results": results,
    }

    with open(OUTPUT_FILE, "w") as fp:
        json.dump(output, fp, indent=2, ensure_ascii=False)
        fp.write("\n")

    logger.info(f"Results saved to {OUTPUT_FILE}")

    print_summary_table(results)


def print_summary_table(results: dict) -> None:
    """Print side-by-side flat-vs-split MINEA scores per model, all metrics."""
    table = PrettyTable()
    table.field_names = [
        "Model",
        "N",
        "F.exact", "S.exact",
        "F.kw",    "S.kw",
        "F.sem",   "S.sem",
        "F.judge", "S.judge",
        "F.any",   "S.any",
    ]
    table.float_format = ".1"

    rows = []
    for model_name, data in results.items():
        flat = data.get("flat", {})
        split = data.get("split", {})
        if "error" in flat or "error" in split:
            continue
        rows.append((
            model_name,
            flat.get("total_needles", 0),
            flat["mean_minea_exact"] * 100,    split["mean_minea_exact"] * 100,
            flat["mean_minea_keyword"] * 100,  split["mean_minea_keyword"] * 100,
            flat["mean_minea_semantic"] * 100, split["mean_minea_semantic"] * 100,
            flat["mean_minea_judge"] * 100,    split["mean_minea_judge"] * 100,
            flat["mean_minea_any"] * 100,      split["mean_minea_any"] * 100,
        ))

    # Sort by flat_judge (column index 8) descending — leaderboard form on
    # the easy condition, so the eye reads right to see the gap.
    rows.sort(key=lambda r: r[8], reverse=True)
    for row in rows:
        table.add_row(row)

    print("\n" + "=" * 100)
    print("MINEA EVALUATION — COREFERENCE-SPLIT (flat vs split, all metrics)")
    print("=" * 100)
    print(table)
    print(
        "\nF.* = flat condition, S.* = split condition. "
        "Flat-vs-split gap shows the difficulty effect.\n"
        "Note: semantic uses canonical (flat) sentence for embedding in BOTH "
        "conditions — drops there reflect extractor failure to produce "
        "semantically-equivalent triples, not surface-area mismatch."
    )
    print()


if __name__ == "__main__":
    asyncio.run(run_experiment())
