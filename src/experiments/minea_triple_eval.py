"""
Experiment: MINEA (Multiple Infused Needle Extraction Accuracy) evaluation
for triple extraction quality across Bedrock frontier models.

Inspired by Seitl et al. (2024) "Assessing the quality of information extraction"
https://arxiv.org/abs/2404.04068

This experiment measures information capture/completeness in triple extraction by:
  1. Generating synthetic "needle" triples contextually relevant to source documents
  2. Injecting needles as natural language into source chunks
  3. Running triple extraction with each candidate model
  4. Measuring extraction success via multiple identification criteria:
     - Exact match (subject/predicate/object all match)
     - Semantic similarity (embedding-based)
     - Keyword overlap (S/O terms present in extracted triples)
     - LLM-as-judge (contextual matching)
  5. Computing MINEA score = # successfully extracted needles / # total needles

Unlike judge-based completeness metrics, MINEA provides:
  - Objective ground truth (we know what should be extracted)
  - Recall-focused measurement (what % of information is captured)
  - No manual annotation required

Usage:
    cd /home/ubuntu/X-RAG/src
    PYTHONPATH=. python -m experiments.minea_triple_eval_refactored
"""

import asyncio
import json
import logging
from statistics import mean

import dspy
from dspy.adapters.chat_adapter import ChatAdapter
from prettytable import PrettyTable

from xrag.config import config
from xrag.paths import CACHE_DIR, DATASETS_DIR, SRC

from experiments.minea.needles import generate_needles, inject_needles
from experiments.minea.extraction import run_extraction_on_enriched_chunk
from experiments.minea.evaluation import evaluate_needle_capture
from experiments.minea.data_loader import load_sample_documents

# ── Configuration ────────────────────────────────────────────────────────────

CANDIDATE_MODELS: dict[str, str] = {
    "opus-4.5": "bedrock/us.anthropic.claude-opus-4-5-20251101-v1:0",
    "sonnet-4": "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0",
    "nova-pro": "bedrock/us.amazon.nova-pro-v1:0",
    "nova-lite": "bedrock/us.amazon.nova-lite-v1:0",
    "mistral-large2": "bedrock/mistral.mistral-large-2407-v1:0",
    "deepseek-v3": "bedrock/deepseek.v3-v1:0",
    "deepseek-v4-pro": "openrouter/deepseek/deepseek-v4-pro",
    "deepseek-v4-flash": "openrouter/deepseek/deepseek-v4-flash",
}

NEEDLE_GEN_MODEL = "bedrock/us.anthropic.claude-opus-4-5-20251101-v1:0"
JUDGE_MODEL = "bedrock/us.anthropic.claude-opus-4-5-20251101-v1:0"
EMBED_MODEL = config.models["embed"]  # Titan v2

SAMPLE_SIZE = 10  # number of documents to sample
NEEDLES_PER_DOC = 5  # number of needle triples per document
SEED = 42
DATASET = "OURS"

# Retry and model configuration
INITIAL_DELAY = config.llm_retry["initial_delay"]
MAX_ATTEMPTS = config.llm_retry["max_attempts"]
MIN_CHUNK_TOKENS = config.preprocess["min_chunk_tokens_thresh"]

# Evaluation thresholds
EMBED_BATCH_LIMIT = 25
KEYWORD_MATCH_THRESHOLD = 0.5  # fraction of keywords that must match
SEMANTIC_SIM_THRESHOLD = 0.75  # cosine similarity threshold for semantic match

# Needle injection and content filtering
ESTIMATED_NEEDLE_LENGTH = 100  # chars, conservative estimate for adaptive needle generation
MAX_CONCURRENT_JUDGE_CALLS = 2  # limit to avoid rate limits and API costs
MIN_CONTENT_LENGTH = 100  # chars, minimum for valid triple extraction
MAX_AVG_LINE_LEN_TOC = 40  # chars, distinguishes ToC from narrative text
CITATION_DENSITY_THRESHOLD = 3.0  # citations per 100 chars indicates reference list
NIAH_EPSILON = 0.001  # tolerance for 10-30% fraction boundary checks

OUTPUT_DIR = SRC / "experiments"
OUTPUT_FILE = OUTPUT_DIR / "minea_results.json"

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)


# ── Main Experiment ──────────────────────────────────────────────────────────


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

    logger.info(
        f"Generating {NEEDLES_PER_DOC} needle triples per document "
        f"({len(documents)} docs)"
    )

    # generate needles for each document (adaptive count for NIAH compliance)
    doc_needles = []
    for doc in documents:
        try:
            # Calculate how many needles we need to reach ~10-30% fraction
            doc_len = len(doc["text"])
            avg_needle_len = ESTIMATED_NEEDLE_LENGTH
            min_needle_len_total = doc_len * 0.1 / (1 - 0.1)  # For 10% fraction
            min_needles_needed = max(NEEDLES_PER_DOC, int(min_needle_len_total / avg_needle_len) + 1)

            # Cap at reasonable maximum (don't generate 100+ needles)
            n_to_generate = min(min_needles_needed, NEEDLES_PER_DOC * 3)

            needles = await generate_needles(
                doc["text"], n_to_generate, NEEDLE_GEN_MODEL, INITIAL_DELAY, MAX_ATTEMPTS
            )
            doc_needles.append(needles)
            logger.info(
                f"Generated {len(needles)} needles for {doc['source_file'][:40]} "
                f"(doc_len={doc_len}, min_needed={min_needles_needed})"
            )
        except Exception as e:
            logger.error(f"Failed to generate needles for {doc['source_file']}: {e}")
            doc_needles.append([])

    # inject needles into documents (with strict 10-30% enforcement)
    if len(documents) != len(doc_needles):
        logger.error(
            f"Length mismatch: {len(documents)} documents != {len(doc_needles)} needle lists. "
            "Some needle generations may have failed."
        )

    enriched_docs = []
    for doc, needles in zip(documents, doc_needles):
        # Inject needles with strict NIAH compliance (10-30% range)
        enriched_text, needle_frac, n_injected = inject_needles(
            chunk_text=doc["text"],
            needles=needles,
            min_frac=0.1,
            max_frac=0.3,
            niah_epsilon=NIAH_EPSILON,
            seed=SEED,
        )

        # Only use the needles that were actually injected (may be subset for compliance)
        injected_needles = needles[:n_injected]

        enriched_docs.append({
            "source_file": doc["source_file"],
            "original_text": doc["text"],
            "enriched_text": enriched_text,
            "needles": injected_needles,  # Store only injected needles
            "needle_fraction": needle_frac,
            "n_needles_generated": len(needles),
            "n_needles_injected": n_injected,
        })

    # run extraction with each model
    results = {}
    for model_name, model_id in CANDIDATE_MODELS.items():
        logger.info(f"{'=' * 60}")
        logger.info(f"Running extraction with {model_name}")
        logger.info(f"{'=' * 60}")

        model_results = []
        for enriched_doc in enriched_docs:
            extraction = await run_extraction_on_enriched_chunk(
                model_name=model_name,
                model_id=model_id,
                original_text=enriched_doc["original_text"],
                enriched_text=enriched_doc["enriched_text"],
                initial_delay=INITIAL_DELAY,
                max_attempts=MAX_ATTEMPTS,
            )

            # Check for extraction failure
            if "error" in extraction:
                logger.error(f"[{model_name}] Extraction failed: {extraction['error']}")
                model_results.append({
                    "source_file": enriched_doc["source_file"],
                    "extraction_failed": True,
                    "error": extraction["error"],
                })
                continue

            # Evaluate needle capture (extraction succeeded)
            try:
                eval_result = await evaluate_needle_capture(
                    needles=enriched_doc["needles"],
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
                model_results.append({
                    "source_file": enriched_doc["source_file"],
                    "extraction": extraction,
                    "evaluation": eval_result,
                })

                logger.info(
                    f"[{model_name}] {enriched_doc['source_file'][:40]}: "
                    f"{extraction['triple_count']} triples, "
                    f"MINEA={eval_result['minea_any']:.2f}"
                )

            except Exception as e:
                logger.error(
                    f"[{model_name}] Evaluation failed on {enriched_doc['source_file']}: {e}"
                )
                model_results.append({
                    "source_file": enriched_doc["source_file"],
                    "evaluation_failed": True,
                    "error": str(e),
                })

        # aggregate per model
        valid_evals = [
            r["evaluation"]
            for r in model_results
            if "evaluation" in r and r["evaluation"]["n_needles"] > 0
        ]

        if valid_evals:
            aggregate = {
                "n_documents": len(model_results),
                "n_valid_evals": len(valid_evals),
                "total_needles": sum(e["n_needles"] for e in valid_evals),
            }
            # Compute mean MINEA scores for each criterion
            for metric in ["exact", "keyword", "semantic", "judge", "any"]:
                aggregate[f"mean_minea_{metric}"] = round(
                    mean(e[f"minea_{metric}"] for e in valid_evals), 4
                )
        else:
            aggregate = {"error": "No valid evaluations"}

        results[model_name] = {
            "model_id": model_id,
            "aggregate": aggregate,
            "per_document": model_results,
        }

        logger.info(
            f"[{model_name}] aggregate MINEA: "
            f"exact={aggregate.get('mean_minea_exact', 0):.2%}, "
            f"keyword={aggregate.get('mean_minea_keyword', 0):.2%}, "
            f"semantic={aggregate.get('mean_minea_semantic', 0):.2%}, "
            f"judge={aggregate.get('mean_minea_judge', 0):.2%}, "
            f"any={aggregate.get('mean_minea_any', 0):.2%}"
        )

    # Compute aggregate needle fraction statistics for NIAH compliance
    needle_fracs = [doc.get("needle_fraction", 0) for doc in enriched_docs]
    if needle_fracs:
        mean_frac = mean(needle_fracs)
        min_frac = min(needle_fracs)
        max_frac = max(needle_fracs)
        # Check if ALL documents are within range (strict compliance)
        all_in_range = all(0.1 - NIAH_EPSILON <= f <= 0.3 + NIAH_EPSILON for f in needle_fracs)
        violations = sum(1 for f in needle_fracs if f < 0.1 - NIAH_EPSILON or f > 0.3 + NIAH_EPSILON)

        logger.info(
            f"Needle fraction: mean={mean_frac:.1%}, "
            f"min={min_frac:.1%}, max={max_frac:.1%}"
        )
        if all_in_range:
            logger.info(f"✓ All {len(needle_fracs)} documents within NIAH 10-30% range")
        else:
            logger.warning(f"⚠️ {violations}/{len(needle_fracs)} documents outside NIAH 10-30% range")

        niah_compliant = all_in_range
    else:
        mean_frac = 0.0
        min_frac = 0.0
        max_frac = 0.0
        niah_compliant = False
        violations = 0

    # save results
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Calculate needle injection statistics
    total_generated = sum(doc.get("n_needles_generated", 0) for doc in enriched_docs)
    total_injected = sum(doc.get("n_needles_injected", 0) for doc in enriched_docs)

    output = {
        "metadata": {
            "dataset": DATASET,
            "sample_size": SAMPLE_SIZE,
            "needles_per_doc": NEEDLES_PER_DOC,
            "needles_generated_total": total_generated,
            "needles_injected_total": total_injected,
            "needles_injected_per_doc_avg": total_injected / len(enriched_docs) if enriched_docs else 0,
            "seed": SEED,
            "needle_gen_model": NEEDLE_GEN_MODEL,
            "judge_model": JUDGE_MODEL,
            "embed_model": EMBED_MODEL,
            "keyword_match_threshold": KEYWORD_MATCH_THRESHOLD,
            "semantic_sim_threshold": SEMANTIC_SIM_THRESHOLD,
            "needle_fraction_mean": mean_frac,
            "needle_fraction_range": [min_frac, max_frac],
            "needle_fraction_violations": violations,
            "niah_compliance_strict": niah_compliant,
        },
        "results": results,
    }

    with open(OUTPUT_FILE, "w") as fp:
        json.dump(output, fp, indent=2, ensure_ascii=False)
        fp.write("\n")

    logger.info(f"Results saved to {OUTPUT_FILE}")

    # print summary table
    print_summary_table(results)


def print_summary_table(results: dict) -> None:
    """Print ranked summary of MINEA scores across models."""
    table = PrettyTable()
    table.field_names = [
        "Model",
        "N_docs",
        "N_needles",
        "MINEA_exact",
        "MINEA_keyword",
        "MINEA_semantic",
        "MINEA_judge",
        "MINEA_any",
    ]
    table.float_format = ".2"

    rows = []
    for model_name, data in results.items():
        agg = data.get("aggregate", {})
        if "error" in agg:
            continue
        rows.append(
            (
                model_name,
                agg["n_valid_evals"],
                agg["total_needles"],
                agg["mean_minea_exact"] * 100,
                agg["mean_minea_keyword"] * 100,
                agg["mean_minea_semantic"] * 100,
                agg["mean_minea_judge"] * 100,
                agg["mean_minea_any"] * 100,
            )
        )

    # sort by MINEA_any (most comprehensive metric)
    rows.sort(key=lambda r: r[6], reverse=True)

    for row in rows:
        table.add_row(row)

    print("\n" + "=" * 80)
    print("MINEA EVALUATION — TRIPLE EXTRACTION MODEL COMPARISON")
    print("=" * 80)
    print(table)
    print("\nMINEA Metrics:")
    print("  - exact: Exact S/P/O match")
    print("  - keyword: Keyword overlap (≥50% of needle keywords found)")
    print("  - semantic: Embedding similarity (≥0.75 cosine)")
    print("  - judge: LLM judge determines semantic equivalence")
    print("  - any: Union of all criteria (most comprehensive)")
    print()


if __name__ == "__main__":
    asyncio.run(run_experiment())
