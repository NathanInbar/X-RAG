"""
Experiment: Compare frontier Bedrock models on triplet extraction quality.

Samples documents from the OURS dataset, runs SPO triple extraction with
each candidate model, then evaluates quality using four metric categories:
  1. Embedding-based topical alignment & coverage (Titan v2)
  2. Keyword overlap (YAKE keyword extraction)
  3. Structural heuristics (parse rate, redundancy, field length)
  4. LLM-as-judge (Opus 4.5)

Note: The LLM judge (Opus 4.5) is also a candidate model. This is standard
practice (judge sees only text + triples, not which model produced them), but
self-preference bias should be noted when reporting results.

Usage:
    cd /home/nathan/Projects/X-RAG/src
    source .venv/bin/activate
    PYTHONPATH=. python -m experiments.triplet_model_comparison
"""

import asyncio
import json
import logging
import random
import re
import time
from statistics import mean, stdev

import dspy
import litellm
import numpy as np
import yake
from prettytable import PrettyTable
from tokenizers import Tokenizer

from xrag.config import config
from xrag.dataset_processing.preprocess import (
    TextSegmenter,
    _ExtractTriples,
)
from xrag.paths import DATASETS_DIR, SRC
from xrag.utils import stable_id_hex

# ── constants ────────────────────────────────────────────────────────────────

CANDIDATE_MODELS: dict[str, str] = {
    "opus-4.5":       "bedrock/us.anthropic.claude-opus-4-5-20251101-v1:0",
    "sonnet-4":       "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0",
    "nova-pro":       "bedrock/us.amazon.nova-pro-v1:0",
    "nova-lite":      "bedrock/us.amazon.nova-lite-v1:0",
    "mistral-large2": "bedrock/mistral.mistral-large-2407-v1:0",
    "deepseek-v3":    "bedrock/deepseek.v3-v1:0",
}

JUDGE_MODEL = "bedrock/us.anthropic.claude-opus-4-5-20251101-v1:0"
EMBED_MODEL = config.models["embed"]           # Titan v2
TOKENIZER_MODEL = config.models["tokenizer"]   # gpt2
SEGMENTER_MODEL = config.models["segmenter"]   # SaT

SAMPLE_SIZE = 10
SEED = 42
MAX_PARALLEL = 4
DATASET = "OURS"

INITIAL_DELAY = config.llm_retry["initial_delay"]
MAX_ATTEMPTS = config.llm_retry["max_attempts"]
MIN_CHUNK_TOKENS = config.preprocess["min_chunk_tokens_thresh"]

EMBED_BATCH_LIMIT = 25  # max texts per Titan v2 embedding request

OUTPUT_DIR = SRC / "experiments"
OUTPUT_FILE = OUTPUT_DIR / "triplet_comparison_results.json"

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)

_tokenizer = Tokenizer.from_pretrained(TOKENIZER_MODEL)

# shared YAKE extractor (initialization loads stopwords, no need to repeat)
_yake_extractor = yake.KeywordExtractor(lan="en", n=2, top=30, dedupLim=0.7)


# ── DSPy judge signature ────────────────────────────────────────────────────

class _JudgeTripleQuality(dspy.Signature):
    """
    You are an expert evaluator assessing the quality of subject-predicate-object
    triples extracted from a source text.  Score each dimension from 1 (worst)
    to 10 (best).

    Faithfulness: Are ALL triples factually supported by the source text?
    Penalize any hallucinated entities or relations not present in the source.

    Completeness: Does the triple set capture all important entities, facts,
    and relationships in the source text?  Penalize significant omissions.

    Granularity: Are triples at an appropriate level of detail?  Penalize
    triples that are too coarse (cramming multiple facts) or too fine
    (trivial / redundant).

    Well-formedness: Are subject, predicate, and object fields clean and
    well-structured?  Penalize sentence fragments, overly long values, or
    ambiguous references.
    """

    source_text: str = dspy.InputField(
        desc="The original text from which triples were extracted"
    )
    triples_json: str = dspy.InputField(
        desc="JSON array of extracted {subject, predicate, object} triples"
    )
    faithfulness: int = dspy.OutputField(desc="Integer score 1-10")
    completeness: int = dspy.OutputField(desc="Integer score 1-10")
    granularity: int = dspy.OutputField(desc="Integer score 1-10")
    well_formedness: int = dspy.OutputField(desc="Integer score 1-10")
    rationale: str = dspy.OutputField(
        desc="Brief explanation justifying the four scores"
    )


# ── helpers ──────────────────────────────────────────────────────────────────

def _parse_judge_score(raw: str | int) -> int:
    """Robustly parse a judge score from DSPy output (may be str or int)."""
    if isinstance(raw, int):
        return max(1, min(10, raw))
    s = str(raw).strip()
    # try direct int conversion first (most common case: "7")
    try:
        return max(1, min(10, int(s)))
    except ValueError:
        pass
    # try float conversion ("8.5" → 9)
    try:
        return max(1, min(10, round(float(s))))
    except ValueError:
        pass
    # handle "N/10" pattern (common LLM output): take the numerator
    m = re.match(r"(\d+)\s*/\s*10", s)
    if m:
        return max(1, min(10, int(m.group(1))))
    # fall back to last integer (handles "Score: 8", "Faithfulness: 9")
    matches = re.findall(r"\d+", s)
    if matches:
        return max(1, min(10, int(matches[-1])))
    raise ValueError(f"Cannot parse score from: {raw!r}")


# ── quantitative metrics ─────────────────────────────────────────────────────

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    d = np.dot(a, b)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(d / n) if n > 0 else 0.0


async def _embed_texts(texts: list[str]) -> list[np.ndarray]:
    """Embed a list of texts via Titan v2 (batched, respecting size limits)."""
    if not texts:
        return []
    all_embeddings: list[np.ndarray] = []
    for i in range(0, len(texts), EMBED_BATCH_LIMIT):
        batch = texts[i : i + EMBED_BATCH_LIMIT]
        delay = INITIAL_DELAY
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                resp = await litellm.aembedding(model=EMBED_MODEL, input=batch)
                # sort by index to guarantee input-order alignment
                sorted_data = sorted(resp.data, key=lambda d: d["index"])
                all_embeddings.extend(np.array(d["embedding"]) for d in sorted_data)
                break
            except Exception as e:
                if attempt < MAX_ATTEMPTS:
                    logger.warning(f"[embed] attempt {attempt} failed: {e}, retrying in {delay}s")
                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    raise
    return all_embeddings


def _split_sentences(text: str) -> list[str]:
    """Rough sentence split on period/newline boundaries."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 10]


def _triple_as_sentence(t: dict) -> str:
    return f"{t['s']} {t['p']} {t['o']}"


async def _embed_triples_and_sentences(
    triples: list[dict], source_text: str
) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    """Embed triples and source sentences in a single API call.
    Returns (triple_embeddings, sentence_embeddings, sentences)."""
    sentences = _split_sentences(source_text)
    if not sentences:
        return [], [], []
    triple_strs = [_triple_as_sentence(t) for t in triples]
    all_texts = triple_strs + sentences
    embeddings = await _embed_texts(all_texts)
    if len(embeddings) != len(all_texts):
        raise ValueError(
            f"Embedding count mismatch: expected {len(all_texts)}, got {len(embeddings)}"
        )
    t_embs = embeddings[: len(triple_strs)]
    s_embs = embeddings[len(triple_strs) :]
    return t_embs, s_embs, sentences


def compute_embed_topical_alignment(
    t_embs: list[np.ndarray], s_embs: list[np.ndarray]
) -> float:
    """Average max cosine similarity of each triple to source sentences."""
    if not t_embs or not s_embs:
        return 0.0
    scores = []
    for te in t_embs:
        best = max(_cosine_sim(te, se) for se in s_embs)
        scores.append(best)
    return mean(scores)


def compute_embed_topical_coverage(
    t_embs: list[np.ndarray], s_embs: list[np.ndarray]
) -> float:
    """Average max cosine similarity of each source sentence to any triple."""
    if not t_embs or not s_embs:
        return 0.0
    scores = []
    for se in s_embs:
        best = max(_cosine_sim(se, te) for te in t_embs)
        scores.append(best)
    return mean(scores)


def calc_entity_recall(triples: list[dict], source_text: str) -> float:
    """Fraction of YAKE-extracted keywords found in triple S/O fields (word-boundary match)."""
    keywords = _yake_extractor.extract_keywords(source_text)
    if not keywords:
        return 0.0
    kw_set = {kw.lower() for kw, _ in keywords}

    triple_text = " ".join(
        f"{t['s']} {t['o']}" for t in triples
    ).lower()

    # word-boundary match to avoid partial-word false positives
    found = sum(
        1 for kw in kw_set
        if re.search(r"\b" + re.escape(kw) + r"\b", triple_text)
    )
    return found / len(kw_set)


def calc_redundancy_ratio(triples: list[dict]) -> float:
    """Fraction of triples that are near-duplicates (Jaccard > 0.8)."""
    if len(triples) <= 1:
        return 0.0

    def _tokens(t: dict) -> set[str]:
        return set(f"{t['s']} {t['p']} {t['o']}".lower().split())

    token_sets = [_tokens(t) for t in triples]
    # mark each triple that has at least one near-duplicate
    is_dup = [False] * len(triples)
    for i in range(len(token_sets)):
        for j in range(i + 1, len(token_sets)):
            inter = len(token_sets[i] & token_sets[j])
            union = len(token_sets[i] | token_sets[j])
            if union > 0 and inter / union > 0.8:
                is_dup[i] = True
                is_dup[j] = True
    return sum(is_dup) / len(triples)


def calc_mean_field_lengths(triples: list[dict]) -> dict[str, float]:
    """Mean token length of S, P, O fields."""
    if not triples:
        return {"s": 0.0, "p": 0.0, "o": 0.0}
    return {
        "s": mean(len(t["s"].split()) for t in triples),
        "p": mean(len(t["p"].split()) for t in triples),
        "o": mean(len(t["o"].split()) for t in triples),
    }


# ── data loading ─────────────────────────────────────────────────────────────

def load_sample_chunks() -> list[dict]:
    """Sample documents from OURS and segment into chunks."""
    dataset_dir = DATASETS_DIR / DATASET
    all_files = sorted(dataset_dir.glob("*.json"))
    if not all_files:
        raise RuntimeError(f"No JSON files found in {dataset_dir}")

    rng = random.Random(SEED)
    sampled = rng.sample(all_files, min(SAMPLE_SIZE, len(all_files)))

    TextSegmenter.configure(model=SEGMENTER_MODEL)

    chunks: list[dict] = []
    for f in sampled:
        with open(f) as fp:
            data = json.load(fp)
        if "essay" not in data:
            logger.warning(f"Skipping {f.name}: no 'essay' key")
            continue
        segments = TextSegmenter.create_segments(data["essay"])
        for seg in segments:
            seg = seg.strip()
            enc = _tokenizer.encode(seg)
            if len(enc) < MIN_CHUNK_TOKENS:
                continue
            chunks.append({
                "id": stable_id_hex(seg),
                "raw_text": seg,
                "approx_n_tokens": len(enc),
                "source_file": f.name,
            })

    logger.info(f"Loaded {len(chunks)} chunks from {len(sampled)} documents")
    return chunks


# ── extraction ───────────────────────────────────────────────────────────────

async def run_extraction(
    model_name: str, model_id: str, chunks: list[dict]
) -> list[dict]:
    """Run triplet extraction for a single model across all chunks."""
    sem = asyncio.Semaphore(MAX_PARALLEL)
    lm = dspy.LM(model_id, max_tokens=16000)
    predict = dspy.Predict(_ExtractTriples)

    async def _extract_one(chunk: dict) -> dict:
        t0 = time.monotonic()
        triples: list[dict] = []
        parse_success = False
        raw_output = ""
        attempts_used = 0

        async with sem:
            delay = INITIAL_DELAY
            for attempt in range(1, MAX_ATTEMPTS + 1):
                attempts_used = attempt
                triples = []  # reset on each attempt
                t_infer = time.monotonic()
                try:
                    with dspy.context(lm=lm):
                        pred = await predict.acall(source_text=chunk["raw_text"])
                    raw_output = getattr(pred, "triples_json", "") or "[]"
                    data = json.loads(raw_output)
                    if not isinstance(data, list):
                        raise ValueError("not a list")
                    for item in data:
                        if not isinstance(item, dict):
                            continue
                        s = (item.get("subject") or "").strip()
                        p = (item.get("predicate") or "").strip()
                        o = (item.get("object") or "").strip()
                        if s and p and o:
                            triples.append({"s": s, "p": p, "o": o})
                    parse_success = True
                    break
                except Exception as e:
                    if attempt < MAX_ATTEMPTS:
                        logger.warning(
                            f"[{model_name}] attempt {attempt} failed: {e}, "
                            f"retrying in {delay}s"
                        )
                        await asyncio.sleep(delay)
                        delay *= 2
                    else:
                        logger.error(
                            f"[{model_name}] all {MAX_ATTEMPTS} attempts failed "
                            f"for chunk {chunk['id'][:8]}: {e}"
                        )

        inference_elapsed = time.monotonic() - t_infer if parse_success else 0.0
        wall_elapsed = time.monotonic() - t0
        return {
            "chunk_id": chunk["id"],
            "source_file": chunk["source_file"],
            "input_tokens": chunk["approx_n_tokens"],
            "triple_count": len(triples),
            "triples": triples,
            "raw_output": raw_output,
            "parse_success": parse_success,
            "has_triples": parse_success and len(triples) > 0,
            "attempts": attempts_used,
            "inference_seconds": round(inference_elapsed, 3),
            "wall_seconds": round(wall_elapsed, 3),
        }

    tasks = [asyncio.create_task(_extract_one(c)) for c in chunks]
    results = await asyncio.gather(*tasks)
    return list(results)


# ── judge ────────────────────────────────────────────────────────────────────

async def judge_extractions(
    chunks: list[dict], extraction_results: list[dict]
) -> None:
    """Run LLM judge on each extraction result (mutates results in place)."""
    judge_lm = dspy.LM(JUDGE_MODEL)
    judge_predict = dspy.Predict(_JudgeTripleQuality)
    sem = asyncio.Semaphore(MAX_PARALLEL)

    async def _judge_one(chunk: dict, result: dict) -> None:
        if not result["has_triples"]:
            result.update(
                judge_faithfulness=0, judge_completeness=0,
                judge_granularity=0, judge_well_formedness=0,
                judge_rationale="No triples extracted",
            )
            return

        triples_for_judge = [
            {"subject": t["s"], "predicate": t["p"], "object": t["o"]}
            for t in result["triples"]
        ]
        triples_str = json.dumps(triples_for_judge, ensure_ascii=False)

        async with sem:
            delay = INITIAL_DELAY
            for attempt in range(1, MAX_ATTEMPTS + 1):
                try:
                    with dspy.context(lm=judge_lm):
                        pred = await judge_predict.acall(
                            source_text=chunk["raw_text"],
                            triples_json=triples_str,
                        )
                    result.update(
                        judge_faithfulness=_parse_judge_score(pred.faithfulness),
                        judge_completeness=_parse_judge_score(pred.completeness),
                        judge_granularity=_parse_judge_score(pred.granularity),
                        judge_well_formedness=_parse_judge_score(pred.well_formedness),
                        judge_rationale=str(pred.rationale),
                    )
                    return
                except Exception as e:
                    if attempt < MAX_ATTEMPTS:
                        logger.warning(
                            f"[judge] attempt {attempt} failed for "
                            f"chunk {result['chunk_id'][:8]}: {e}"
                        )
                        await asyncio.sleep(delay)
                        delay *= 2
                    else:
                        logger.error(
                            f"[judge] all attempts failed for "
                            f"chunk {result['chunk_id'][:8]}: {e}"
                        )
                        result.update(
                            judge_faithfulness=None, judge_completeness=None,
                            judge_granularity=None, judge_well_formedness=None,
                            judge_rationale=f"Judge error: {e}",
                        )

    tasks = [
        asyncio.create_task(_judge_one(c, r))
        for c, r in zip(chunks, extraction_results)
    ]
    await asyncio.gather(*tasks)


# ── quantitative scoring ─────────────────────────────────────────────────────

async def compute_quant_metrics(
    chunks: list[dict], extraction_results: list[dict]
) -> None:
    """Compute embedding-based and heuristic metrics (mutates results)."""
    embed_sem = asyncio.Semaphore(MAX_PARALLEL)

    async def _score_one(chunk: dict, result: dict) -> None:
        triples = result["triples"]
        text = chunk["raw_text"]

        zero_metrics = dict(
            topical_alignment=0.0, topical_coverage=0.0,
            topical_coverage_per_triple=0.0,
            keyword_overlap=0.0, redundancy_ratio=0.0,
            mean_field_lengths={"s": 0.0, "p": 0.0, "o": 0.0},
        )

        if not triples:
            result.update(zero_metrics)
            return

        try:
            # single embedding call for both alignment and coverage
            async with embed_sem:
                t_embs, s_embs, _ = await _embed_triples_and_sentences(triples, text)

            alignment = compute_embed_topical_alignment(t_embs, s_embs)
            coverage = compute_embed_topical_coverage(t_embs, s_embs)

            result.update(
                topical_alignment=round(alignment, 4),
                topical_coverage=round(coverage, 4),
                topical_coverage_per_triple=round(coverage / len(triples), 4),
                keyword_overlap=round(calc_entity_recall(triples, text), 4),
                redundancy_ratio=round(calc_redundancy_ratio(triples), 4),
                mean_field_lengths=calc_mean_field_lengths(triples),
            )
        except Exception as e:
            logger.error(
                f"[quant] metrics failed for chunk {result['chunk_id'][:8]}: {e}"
            )
            result.update(zero_metrics)

    tasks = [
        asyncio.create_task(_score_one(c, r))
        for c, r in zip(chunks, extraction_results)
    ]
    await asyncio.gather(*tasks)


# ── aggregation & output ─────────────────────────────────────────────────────

def _safe_mean(values: list[float]) -> float:
    return round(mean(values), 4) if values else 0.0


def _safe_stdev(values: list[float]) -> float:
    return round(stdev(values), 4) if len(values) >= 2 else 0.0


def aggregate(results: list[dict], common_chunk_ids: set[str] | None = None) -> dict:
    """Compute per-model aggregate statistics.

    If common_chunk_ids is provided, quality metrics are computed only over
    chunks in that set (the intersection where ALL models produced triples).
    Otherwise, quality metrics use all chunks with has_triples=True.

    Triple count, latency, and parse rates always use all results.
    """
    n = len(results)
    if common_chunk_ids is not None:
        quality_set = [
            r for r in results
            if r.get("has_triples") and r["chunk_id"] in common_chunk_ids
        ]
    else:
        quality_set = [r for r in results if r.get("has_triples")]

    def _vals(key: str) -> list[float]:
        return [r[key] for r in quality_set if key in r and r[key] is not None]

    parse_count = sum(1 for r in results if r["parse_success"])
    triples_count = sum(1 for r in results if r.get("has_triples"))

    return {
        "n_chunks": n,
        "n_with_triples": triples_count,
        "n_quality_eval": len(quality_set),
        "parse_success_rate": round(parse_count / n, 4) if n else 0,
        "has_triples_rate": round(triples_count / n, 4) if n else 0,
        "mean_triple_count": _safe_mean([r["triple_count"] for r in results]),
        "mean_inference_seconds": _safe_mean([r["inference_seconds"] for r in results]),
        "mean_wall_seconds": _safe_mean([r["wall_seconds"] for r in results]),
        "mean_attempts": _safe_mean([r["attempts"] for r in results]),
        # embedding metrics (quality set only)
        "mean_topical_alignment": _safe_mean(_vals("topical_alignment")),
        "mean_topical_coverage": _safe_mean(_vals("topical_coverage")),
        "mean_topical_coverage_per_triple": _safe_mean(_vals("topical_coverage_per_triple")),
        # keyword overlap
        "mean_keyword_overlap": _safe_mean(_vals("keyword_overlap")),
        # structural
        "mean_redundancy_ratio": _safe_mean(_vals("redundancy_ratio")),
        # judge (quality set only)
        "mean_judge_faithfulness": _safe_mean(_vals("judge_faithfulness")),
        "mean_judge_completeness": _safe_mean(_vals("judge_completeness")),
        "mean_judge_granularity": _safe_mean(_vals("judge_granularity")),
        "mean_judge_well_formedness": _safe_mean(_vals("judge_well_formedness")),
        # chunk length covariate (for analysing quality vs chunk size)
        "mean_input_tokens": _safe_mean([r["input_tokens"] for r in quality_set]),
        "std_input_tokens": _safe_stdev([r["input_tokens"] for r in quality_set]),
        "min_input_tokens": min((r["input_tokens"] for r in quality_set), default=0),
        "max_input_tokens": max((r["input_tokens"] for r in quality_set), default=0),
        # stdev for key metrics
        "std_topical_alignment": _safe_stdev(_vals("topical_alignment")),
        "std_topical_coverage": _safe_stdev(_vals("topical_coverage")),
        "std_judge_faithfulness": _safe_stdev(_vals("judge_faithfulness")),
        "std_judge_completeness": _safe_stdev(_vals("judge_completeness")),
    }


def print_summary(all_results: dict) -> None:
    """Print a ranked summary table."""
    table = PrettyTable()
    table.field_names = [
        "Model", "Parse%", "Triple%", "#Tri", "Infer(s)",
        "Align", "Cov", "Cov/Tri", "KwOvlp", "Redund",
        "JFaith", "JCompl", "JGran", "JWForm",
    ]
    table.float_format = ".2"

    rows = []
    for name, data in all_results.items():
        agg = data["aggregate"]
        rows.append((
            name,
            agg["parse_success_rate"] * 100,
            agg["has_triples_rate"] * 100,
            agg["mean_triple_count"],
            agg["mean_inference_seconds"],
            agg["mean_topical_alignment"],
            agg["mean_topical_coverage"],
            agg["mean_topical_coverage_per_triple"],
            agg["mean_keyword_overlap"],
            agg["mean_redundancy_ratio"],
            agg["mean_judge_faithfulness"],
            agg["mean_judge_completeness"],
            agg["mean_judge_granularity"],
            agg["mean_judge_well_formedness"],
        ))

    # sort by judge faithfulness (most defensible single metric)
    rows.sort(key=lambda r: r[10], reverse=True)

    for row in rows:
        table.add_row(row)

    print("\n" + "=" * 80)
    print("TRIPLET EXTRACTION MODEL COMPARISON — SUMMARY")
    print("=" * 80)
    print(table)
    print()


# ── main ─────────────────────────────────────────────────────────────────────

async def main() -> None:
    logger.info(f"Loading sample chunks from {DATASET} (n={SAMPLE_SIZE}, seed={SEED})")
    chunks = load_sample_chunks()
    if not chunks:
        logger.error("No chunks loaded — aborting")
        return

    metadata = {
        "dataset": DATASET,
        "sample_size": SAMPLE_SIZE,
        "seed": SEED,
        "n_chunks": len(chunks),
        "embed_model": EMBED_MODEL,
        "judge_model": JUDGE_MODEL,
        "min_chunk_tokens": MIN_CHUNK_TOKENS,
        "max_parallel": MAX_PARALLEL,
        "max_attempts": MAX_ATTEMPTS,
        "extract_signature": _ExtractTriples.__doc__.strip(),
    }

    # raw_results stores per-chunk data before intersection aggregation
    raw_results: dict[str, list[dict]] = {}
    output: dict[str, dict] = {"metadata": metadata}
    OUTPUT_DIR.mkdir(exist_ok=True)

    def _save() -> None:
        with open(OUTPUT_FILE, "w") as fp:
            json.dump(output, fp, indent=2, ensure_ascii=False, default=str)

    for model_name, model_id in CANDIDATE_MODELS.items():
        logger.info(f"{'='*60}")
        logger.info(f"Extracting with {model_name} ({model_id})")
        logger.info(f"{'='*60}")

        try:
            extraction_results = await run_extraction(model_name, model_id, chunks)
            parsed = sum(1 for r in extraction_results if r["parse_success"])
            with_triples = sum(1 for r in extraction_results if r["has_triples"])
            logger.info(
                f"[{model_name}] extraction done: {parsed}/{len(chunks)} parsed, "
                f"{with_triples} with triples"
            )

            logger.info(f"[{model_name}] computing quantitative metrics...")
            await compute_quant_metrics(chunks, extraction_results)

            logger.info(f"[{model_name}] running LLM judge...")
            await judge_extractions(chunks, extraction_results)

            raw_results[model_name] = extraction_results

            # preliminary aggregate (without intersection, for incremental logging)
            agg = aggregate(extraction_results)
            output[model_name] = {
                "model_id": model_id,
                "aggregate": agg,
                "per_chunk": extraction_results,
            }

            logger.info(
                f"[{model_name}] done — parse={agg['parse_success_rate']:.0%}, "
                f"triples={agg['has_triples_rate']:.0%}, "
                f"j_faith={agg['mean_judge_faithfulness']:.1f}, "
                f"j_compl={agg['mean_judge_completeness']:.1f}"
            )
        except Exception as e:
            logger.error(f"[{model_name}] FAILED — skipping: {e}")
            output[model_name] = {
                "model_id": model_id,
                "error": str(e),
            }

        _save()

    # recompute aggregates over the common intersection of chunks where
    # ALL models produced triples (eliminates selection bias)
    successful_models = [m for m in raw_results if m in output and "error" not in output[m]]
    if len(successful_models) >= 2:
        per_model_triple_ids = [
            {r["chunk_id"] for r in raw_results[m] if r.get("has_triples")}
            for m in successful_models
        ]
        common_ids = set.intersection(*per_model_triple_ids)
        logger.info(
            f"Common evaluation set: {len(common_ids)} chunks "
            f"(all {len(successful_models)} models produced triples)"
        )
        metadata["n_common_eval_chunks"] = len(common_ids)

        for model_name in successful_models:
            output[model_name]["aggregate"] = aggregate(
                raw_results[model_name], common_chunk_ids=common_ids
            )

        _save()

    logger.info(f"Results written to {OUTPUT_FILE}")

    model_results = {k: v for k, v in output.items() if k != "metadata" and "aggregate" in v}
    if model_results:
        print_summary(model_results)


if __name__ == "__main__":
    asyncio.run(main())
