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
    PYTHONPATH=. python -m experiments.minea_triple_eval
"""

import asyncio
from collections import defaultdict
import json
import logging
import random
import re
import time
from pathlib import Path
from statistics import mean
from typing import Any

import dspy
from dspy.adapters.chat_adapter import ChatAdapter
import litellm
import numpy as np
from prettytable import PrettyTable

from xrag.config import config
from xrag.dataset_processing.preprocess import _ExtractTriples, process_dataset_file
from xrag.paths import CACHE_DIR, DATASETS_DIR, SRC

# ── constants ────────────────────────────────────────────────────────────────

CANDIDATE_MODELS: dict[str, str] = {
    "opus-4.5": "bedrock/us.anthropic.claude-opus-4-5-20251101-v1:0",
    "sonnet-4": "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0",
    "nova-pro": "bedrock/us.amazon.nova-pro-v1:0",
    "nova-lite": "bedrock/us.amazon.nova-lite-v1:0",
    "mistral-large2": "bedrock/mistral.mistral-large-2407-v1:0",
    "deepseek-v3": "bedrock/deepseek.v3-v1:0",
}

NEEDLE_GEN_MODEL = "bedrock/us.anthropic.claude-opus-4-5-20251101-v1:0"
JUDGE_MODEL = "bedrock/us.anthropic.claude-opus-4-5-20251101-v1:0"
EMBED_MODEL = config.models["embed"]  # Titan v2

SAMPLE_SIZE = 10  # number of documents to sample
NEEDLES_PER_DOC = 5  # number of needle triples per document
SEED = 42
MAX_PARALLEL = 4
DATASET = "OURS"

INITIAL_DELAY = config.llm_retry["initial_delay"]
MAX_ATTEMPTS = config.llm_retry["max_attempts"]
MIN_CHUNK_TOKENS = config.preprocess["min_chunk_tokens_thresh"]

EMBED_BATCH_LIMIT = 25
KEYWORD_MATCH_THRESHOLD = 0.5  # fraction of keywords that must match
SEMANTIC_SIM_THRESHOLD = 0.75  # cosine similarity threshold for semantic match

OUTPUT_DIR = SRC / "experiments"
OUTPUT_FILE = OUTPUT_DIR / "minea_results.json"

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)


# ── DSPy signatures ──────────────────────────────────────────────────────────


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


class _JudgeNeedleMatch(dspy.Signature):
    """
    You are an expert at named entity recognition. Given:
    - A needle triple (the ground truth we're looking for)
    - A list of extracted triples from the text

    Determine if the needle was successfully captured by ANY of the extracted triples.

    A needle is considered "captured" if there exists an extracted triple that:
    - Refers to the same subject entity (may use synonyms/variants)
    - Expresses the same or very similar relationship
    - Refers to the same object entity (may use synonyms/variants)

    Be lenient with minor phrasing differences, but strict about semantic equivalence.
    """

    needle_subject: str = dspy.InputField()
    needle_predicate: str = dspy.InputField()
    needle_object: str = dspy.InputField()
    extracted_triples_json: str = dspy.InputField(
        desc="JSON array of extracted {subject, predicate, object} triples"
    )
    is_captured: bool = dspy.OutputField(
        desc="True if needle was captured, False otherwise"
    )
    matched_triple_index: int = dspy.OutputField(
        desc="Index of matching triple in extracted list, or -1 if not found"
    )
    reasoning: str = dspy.OutputField(desc="Brief explanation of the decision")


# ── needle generation ────────────────────────────────────────────────────────


async def generate_needles(
    paper_text: str, n_needles: int
) -> list[dict[str, Any]]:
    """Generate needle triples for a given paper excerpt."""
    lm = dspy.LM(NEEDLE_GEN_MODEL, max_tokens=4096)
    predict = dspy.Predict(_GenerateNeedleTriples)

    # use first ~5000 chars as context for needle generation
    # (matches the chunk size used for injection to ensure contextual relevance)
    excerpt = paper_text[:5000]

    delay = INITIAL_DELAY
    for attempt in range(1, MAX_ATTEMPTS + 1):
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
                # Ensure we have string values before calling .strip()
                s_raw = item.get("subject") or ""
                p_raw = item.get("predicate") or ""
                o_raw = item.get("object") or ""
                sent_raw = item.get("sentence") or ""
                if not all(isinstance(v, str) for v in [s_raw, p_raw, o_raw, sent_raw]):
                    continue
                s = s_raw.strip()
                p = p_raw.strip()
                o = o_raw.strip()
                sent = sent_raw.strip()
                kws = item.get("keywords", [])
                if not (s and p and o and sent):
                    continue
                if not isinstance(kws, list):
                    kws = []
                # Ensure keywords are strings before calling .strip()
                validated_kws = []
                for k in kws:
                    if isinstance(k, str) and k.strip():
                        validated_kws.append(k.strip())
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
            if attempt < MAX_ATTEMPTS:
                logger.warning(
                    f"[needle-gen] attempt {attempt} failed: {e}, retrying in {delay}s"
                )
                await asyncio.sleep(delay)
                delay *= 2
            else:
                logger.error(f"[needle-gen] all attempts failed: {e}")
                return []  # Return empty list instead of raising

    return []  # Should never reach here, but defensive programming


def inject_needles(chunk_text: str, needles: list[dict[str, Any]]) -> tuple[str, float]:
    """
    Inject needle sentences into chunk text at random positions.

    Per Seitl et al. (2024): "We scatter several needles at random over the
    text document body (such that the inserted needles fill 10 to 30% of
    the enriched text)."

    Inserts needles between sentences to maintain natural flow.

    Returns:
        (enriched_text, needle_fraction): enriched text and the fraction of text
        that is needles (0.0-1.0)
    """
    # split into sentences
    sentences = re.split(r"(?<=[.!?])\s+", chunk_text.strip())

    # calculate injection positions (random placement per paper)
    n_needles = len(needles)
    if n_needles == 0:
        return chunk_text, 0.0

    # Select random positions for needle insertion
    # Use seeded RNG for reproducibility
    rng = random.Random(SEED)

    # Map position -> list of needle indices to handle multiple needles per position
    position_to_needles = defaultdict(list)

    if len(sentences) >= n_needles:
        # Sample n_needles positions without replacement
        positions = sorted(rng.sample(range(len(sentences)), n_needles))
        for needle_idx, pos in enumerate(positions):
            position_to_needles[pos].append(needle_idx)
    else:
        # More needles than sentences - some positions will have multiple needles
        # Sample with replacement
        for needle_idx in range(n_needles):
            pos = rng.randint(0, len(sentences) - 1)
            position_to_needles[pos].append(needle_idx)

    # Insert needles at calculated positions
    enriched = []
    for i, sent in enumerate(sentences):
        enriched.append(sent)
        # Insert all needles mapped to this position
        if i in position_to_needles:
            for needle_idx in position_to_needles[i]:
                enriched.append(needles[needle_idx]["sentence"])

    # Handle needles that map to position == len(sentences) (after last sentence)
    # This can happen with randint(0, len(sentences)-1) edge case
    if len(sentences) in position_to_needles:
        for needle_idx in position_to_needles[len(sentences)]:
            enriched.append(needles[needle_idx]["sentence"])

    enriched_text = " ".join(enriched)

    # Calculate needle fraction for NIAH compliance validation
    original_len = len(chunk_text)
    enriched_len = len(enriched_text)
    needle_fraction = (enriched_len - original_len) / enriched_len if enriched_len > 0 else 0.0

    # Validate NIAH requirement (10-30%)
    if not (0.1 <= needle_fraction <= 0.3):
        logger.warning(
            f"Needle fraction {needle_fraction:.1%} outside NIAH recommended range (10-30%)"
        )

    return enriched_text, needle_fraction


# ── extraction ───────────────────────────────────────────────────────────────


async def run_extraction_on_enriched_chunk(
    model_name: str,
    model_id: str,
    original_text: str,
    enriched_text: str,
) -> dict[str, Any]:
    """Run triple extraction on a needle-enriched chunk."""
    try:
        model_info = litellm.get_model_info(model_id)
        model_max = model_info.get("max_output_tokens", 4000)
    except Exception as e:
        logger.error(f"[{model_name}] invalid model ID {model_id}: {e}")
        return {
            "model_name": model_name,
            "model_id": model_id,
            "triple_count": 0,
            "triples": [],
            "parse_success": False,
            "attempts": 0,
            "elapsed_seconds": 0.0,
            "error": f"Invalid model ID: {e}",
        }

    lm = dspy.LM(model_id, max_tokens=min(16000, model_max))
    predict = dspy.Predict(_ExtractTriples)

    t0 = time.monotonic()
    triples: list[dict] = []
    parse_success = False
    raw_output = ""
    attempts_used = 0

    delay = INITIAL_DELAY
    for attempt in range(1, MAX_ATTEMPTS + 1):
        attempts_used = attempt
        triples = []
        try:
            with dspy.context(lm=lm):
                pred = await predict.acall(source_text=enriched_text)
            raw_output = getattr(pred, "triples_json", "") or "[]"
            raw_output = raw_output.strip()

            if raw_output.startswith("```"):
                raw_output = re.sub(r"^```\w*\n?", "", raw_output)
                raw_output = re.sub(r"\n?```\s*$", "", raw_output)
                raw_output = raw_output.strip()

            data = json.loads(raw_output)
            if not isinstance(data, list):
                raise ValueError("not a list")

            for item in data:
                if not isinstance(item, dict):
                    continue
                # Ensure we have string values before calling .strip()
                s_raw = item.get("subject") or ""
                p_raw = item.get("predicate") or ""
                o_raw = item.get("object") or ""
                if not isinstance(s_raw, str) or not isinstance(p_raw, str) or not isinstance(o_raw, str):
                    continue
                s = s_raw.strip()
                p = p_raw.strip()
                o = o_raw.strip()
                if s and p and o:
                    triples.append({"s": s, "p": p, "o": o})

            parse_success = True
            break

        except Exception as e:
            if attempt < MAX_ATTEMPTS:
                logger.warning(
                    f"[{model_name}] attempt {attempt} failed: {e}, retrying in {delay}s"
                )
                await asyncio.sleep(delay)
                delay *= 2
            else:
                logger.error(f"[{model_name}] all attempts failed: {e}")

    elapsed = time.monotonic() - t0
    result = {
        "model_name": model_name,
        "model_id": model_id,
        "triple_count": len(triples),
        "triples": triples,
        "parse_success": parse_success,
        "attempts": attempts_used,
        "elapsed_seconds": round(elapsed, 3),
    }

    # Add error indicator if extraction failed
    if not parse_success:
        result["error"] = "Extraction failed after all retry attempts"

    return result


# ── needle identification ────────────────────────────────────────────────────


def exact_match(needle: dict, extracted: list[dict]) -> bool:
    """Check if needle exactly matches any extracted triple."""
    n_s = needle["subject"].lower().strip()
    n_p = needle["predicate"].lower().strip()
    n_o = needle["object"].lower().strip()

    for t in extracted:
        if (
            t["s"].lower().strip() == n_s
            and t["p"].lower().strip() == n_p
            and t["o"].lower().strip() == n_o
        ):
            return True
    return False


def keyword_match(
    needle: dict, extracted: list[dict], threshold: float = KEYWORD_MATCH_THRESHOLD
) -> bool:
    """
    Check if needle keywords appear in extracted triples.

    A needle is matched if at least `threshold` fraction of its keywords
    appear in any extracted triple's S/P/O fields (word-boundary match).
    """
    keywords = needle.get("keywords", [])
    if not keywords:
        return False

    # normalize keywords
    kw_set = {kw.lower().strip() for kw in keywords}

    for t in extracted:
        triple_text = f"{t['s']} {t['p']} {t['o']}".lower()
        matched = sum(
            1
            for kw in kw_set
            if re.search(r"\b" + re.escape(kw) + r"\b", triple_text)
        )
        if matched / len(kw_set) >= threshold:
            return True

    return False


async def _embed_texts(texts: list[str]) -> list[np.ndarray]:
    """Embed texts via Titan v2 (batched)."""
    if not texts:
        return []

    all_embeddings: list[np.ndarray] = []
    for i in range(0, len(texts), EMBED_BATCH_LIMIT):
        batch = texts[i : i + EMBED_BATCH_LIMIT]
        delay = INITIAL_DELAY
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                resp = await litellm.aembedding(model=EMBED_MODEL, input=batch)
                sorted_data = sorted(resp.data, key=lambda d: d["index"])
                all_embeddings.extend(np.array(d["embedding"]) for d in sorted_data)
                break
            except Exception as e:
                if attempt < MAX_ATTEMPTS:
                    logger.warning(
                        f"[embed] attempt {attempt} failed: {e}, retrying in {delay}s"
                    )
                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    raise
    return all_embeddings


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    d = np.dot(a, b)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(d / n) if n > 0 else 0.0


async def semantic_match(
    needle: dict, extracted: list[dict], threshold: float = SEMANTIC_SIM_THRESHOLD
) -> bool:
    """
    Check if needle is semantically similar to any extracted triple.

    Embeds needle sentence and all extracted triples, returns True if
    max cosine similarity exceeds threshold.
    """
    if not extracted:
        return False

    needle_sent = needle.get("sentence", "")
    if not needle_sent:
        # fall back to concatenating S P O
        needle_sent = f"{needle['subject']} {needle['predicate']} {needle['object']}"

    triple_sents = [f"{t['s']} {t['p']} {t['o']}" for t in extracted]
    all_texts = [needle_sent] + triple_sents

    try:
        embeddings = await _embed_texts(all_texts)
        if len(embeddings) != len(all_texts):
            logger.warning(
                f"Embedding mismatch: expected {len(all_texts)}, got {len(embeddings)}"
            )
            return False

        needle_emb = embeddings[0]
        triple_embs = embeddings[1:]

        if not triple_embs:
            logger.warning("[semantic_match] no triple embeddings to compare")
            return False

        max_sim = max(_cosine_sim(needle_emb, te) for te in triple_embs)
        return max_sim >= threshold

    except Exception as e:
        logger.error(f"[semantic_match] error: {e}")
        return False


async def llm_judge_match(needle: dict, extracted: list[dict]) -> bool:
    """Use LLM judge to determine if needle was captured."""
    if not extracted:
        return False

    lm = dspy.LM(JUDGE_MODEL)
    predict = dspy.Predict(_JudgeNeedleMatch)

    extracted_for_judge = [
        {"subject": t["s"], "predicate": t["p"], "object": t["o"]} for t in extracted
    ]
    extracted_json = json.dumps(extracted_for_judge, ensure_ascii=False)

    delay = INITIAL_DELAY
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with dspy.context(lm=lm):
                pred = await predict.acall(
                    needle_subject=needle["subject"],
                    needle_predicate=needle["predicate"],
                    needle_object=needle["object"],
                    extracted_triples_json=extracted_json,
                )

            is_captured = getattr(pred, "is_captured", False)
            # robust parsing (DSPy may return str instead of bool)
            if isinstance(is_captured, str):
                is_captured = is_captured.strip().lower() in ("true", "yes", "1")

            return bool(is_captured)

        except Exception as e:
            if attempt < MAX_ATTEMPTS:
                logger.warning(
                    f"[judge] attempt {attempt} failed: {e}, retrying in {delay}s"
                )
                await asyncio.sleep(delay)
                delay *= 2
            else:
                logger.error(f"[judge] all attempts failed: {e}")
                return False

    return False


async def evaluate_needle_capture(
    needles: list[dict], extraction_result: dict
) -> dict[str, Any]:
    """
    Evaluate needle capture using multiple criteria.

    Returns dict with per-needle results and aggregate MINEA scores.
    """
    extracted = extraction_result["triples"]
    n_total = len(needles)

    if n_total == 0:
        return {
            "n_needles": 0,
            "minea_exact": 0.0,
            "minea_keyword": 0.0,
            "minea_semantic": 0.0,
            "minea_judge": 0.0,
            "minea_any": 0.0,
            "per_needle": [],
        }

    per_needle = []
    sem_tasks = []

    # synchronous checks (exact, keyword)
    for needle in needles:
        exact = exact_match(needle, extracted)
        keyword = keyword_match(needle, extracted)
        per_needle.append({
            "needle": needle,
            "exact_match": exact,
            "keyword_match": keyword,
            "semantic_match": None,  # will fill async
            "judge_match": None,  # will fill async
        })
        # schedule async semantic check
        sem_tasks.append(semantic_match(needle, extracted))

    # await semantic matches
    sem_results = await asyncio.gather(*sem_tasks)
    for i, sem in enumerate(sem_results):
        per_needle[i]["semantic_match"] = sem

    # await judge matches (run sequentially to avoid rate limits)
    for i, needle in enumerate(needles):
        judge = await llm_judge_match(needle, extracted)
        per_needle[i]["judge_match"] = judge

    # compute MINEA scores
    n_exact = sum(1 for r in per_needle if r["exact_match"])
    n_keyword = sum(1 for r in per_needle if r["keyword_match"])
    n_semantic = sum(1 for r in per_needle if r["semantic_match"])
    n_judge = sum(1 for r in per_needle if r["judge_match"])
    n_any = sum(
        1
        for r in per_needle
        if (
            r["exact_match"]
            or r["keyword_match"]
            or r["semantic_match"]
            or r["judge_match"]
        )
    )

    return {
        "n_needles": n_total,
        "minea_exact": round(n_exact / n_total, 4),
        "minea_keyword": round(n_keyword / n_total, 4),
        "minea_semantic": round(n_semantic / n_total, 4),
        "minea_judge": round(n_judge / n_total, 4),
        "minea_any": round(n_any / n_total, 4),  # union of all criteria
        "per_needle": per_needle,
    }


# ── data loading ─────────────────────────────────────────────────────────────


async def load_sample_documents() -> list[dict]:
    """
    Load chunks using SaT segmentation from preprocess cache.

    Uses the same chunking approach as triplet_model_comparison experiment
    to ensure consistency across evaluations.
    """
    dataset_dir = DATASETS_DIR / DATASET
    all_files = sorted(dataset_dir.glob("*.json"))
    if not all_files:
        raise RuntimeError(f"No JSON files found in {dataset_dir}")

    rng = random.Random(SEED)
    sampled = rng.sample(all_files, min(SAMPLE_SIZE, len(all_files)))

    # Ensure cache exists for every sampled document (generates if missing)
    for f in sampled:
        cache_file = CACHE_DIR / f"{f.stem}__chunks.json"
        if not cache_file.is_file():
            logger.info(f"Cache miss for {f.name} — running preprocess with SaT segmentation (may take a while)...")
            try:
                await process_dataset_file(f)
            except Exception as e:
                logger.error(f"Failed to preprocess {f.name}: {e} — skipping document")
                continue

    # Load chunks from cache
    all_chunks = []
    for f in sampled:
        cache_file = CACHE_DIR / f"{f.stem}__chunks.json"
        if not cache_file.is_file():
            logger.warning(f"No cache file for {f.name}, skipping")
            continue

        with open(cache_file) as fp:
            doc_entries = json.load(fp)

        for doc in doc_entries:
            for chunk in doc["chunks"]:
                if chunk["approx_n_tokens"] < MIN_CHUNK_TOKENS:
                    continue
                all_chunks.append({
                    "source_file": f.name,
                    "text": chunk["raw_text"],
                    "approx_n_tokens": chunk["approx_n_tokens"],
                    "chunk_id": chunk["id"],
                })

    # Group chunks by document and select one chunk per document
    # (We want N documents, not N chunks, to match original experiment design)
    doc_to_chunks = {}
    for chunk in all_chunks:
        source = chunk["source_file"]
        if source not in doc_to_chunks:
            doc_to_chunks[source] = []
        doc_to_chunks[source].append(chunk)

    # Select the longest chunk from each document (most content for needle injection)
    documents = []
    for source_file, chunks in doc_to_chunks.items():
        best_chunk = max(chunks, key=lambda c: c["approx_n_tokens"])
        documents.append({
            "source_file": source_file,
            "text": best_chunk["text"],
            "approx_n_tokens": best_chunk["approx_n_tokens"],
            "chunk_id": best_chunk["chunk_id"],
        })

    logger.info(f"Loaded {len(documents)} chunks from {len(sampled)} documents (using SaT segmentation)")
    return documents


# ── main experiment ──────────────────────────────────────────────────────────


async def run_experiment() -> None:
    dspy.configure(adapter=ChatAdapter())

    logger.info(f"Loading {SAMPLE_SIZE} documents from {DATASET} (seed={SEED})")
    documents = await load_sample_documents()
    if not documents:
        logger.error("No documents loaded — aborting")
        return

    logger.info(
        f"Generating {NEEDLES_PER_DOC} needle triples per document "
        f"({len(documents)} docs)"
    )

    # generate needles for each document
    doc_needles = []
    for doc in documents:
        try:
            needles = await generate_needles(doc["text"], NEEDLES_PER_DOC)
            doc_needles.append(needles)
            logger.info(
                f"Generated {len(needles)} needles for {doc['source_file'][:40]}"
            )
        except Exception as e:
            logger.error(f"Failed to generate needles for {doc['source_file']}: {e}")
            doc_needles.append([])

    # inject needles into documents
    if len(documents) != len(doc_needles):
        logger.error(
            f"Length mismatch: {len(documents)} documents != {len(doc_needles)} needle lists. "
            "Some needle generations may have failed."
        )
        # This should not happen since we append [] on exception, but check defensively

    enriched_docs = []
    for doc, needles in zip(documents, doc_needles):
        enriched_text, needle_frac = inject_needles(doc["text"], needles)
        enriched_docs.append({
            "source_file": doc["source_file"],
            "original_text": doc["text"],
            "enriched_text": enriched_text,
            "needles": needles,
            "needle_fraction": needle_frac,
        })

    # run extraction with each model
    results = {}
    for model_name, model_id in CANDIDATE_MODELS.items():
        logger.info(f"{'=' * 60}")
        logger.info(f"Running extraction with {model_name}")
        logger.info(f"{'=' * 60}")

        model_results = []
        for enriched_doc in enriched_docs:
            try:
                extraction = await run_extraction_on_enriched_chunk(
                    model_name,
                    model_id,
                    enriched_doc["original_text"],
                    enriched_doc["enriched_text"],
                )

                # evaluate needle capture
                eval_result = await evaluate_needle_capture(
                    enriched_doc["needles"], extraction
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
                    f"[{model_name}] failed on {enriched_doc['source_file']}: {e}"
                )
                model_results.append({
                    "source_file": enriched_doc["source_file"],
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
                "mean_minea_exact": round(
                    mean(e["minea_exact"] for e in valid_evals), 4
                ),
                "mean_minea_keyword": round(
                    mean(e["minea_keyword"] for e in valid_evals), 4
                ),
                "mean_minea_semantic": round(
                    mean(e["minea_semantic"] for e in valid_evals), 4
                ),
                "mean_minea_judge": round(
                    mean(e["minea_judge"] for e in valid_evals), 4
                ),
                "mean_minea_any": round(
                    mean(e["minea_any"] for e in valid_evals), 4
                ),
            }
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
        niah_compliant = 0.1 <= mean_frac <= 0.3

        logger.info(
            f"Needle fraction: mean={mean_frac:.1%}, "
            f"min={min_frac:.1%}, max={max_frac:.1%}"
        )
        if not niah_compliant:
            logger.warning(
                f"Mean needle fraction {mean_frac:.1%} outside NIAH range (10-30%). "
                f"Consider adjusting NEEDLES_PER_DOC={NEEDLES_PER_DOC}"
            )
    else:
        mean_frac = 0.0
        min_frac = 0.0
        max_frac = 0.0
        niah_compliant = False

    # save results
    OUTPUT_DIR.mkdir(exist_ok=True)
    output = {
        "metadata": {
            "dataset": DATASET,
            "sample_size": SAMPLE_SIZE,
            "needles_per_doc": NEEDLES_PER_DOC,
            "seed": SEED,
            "needle_gen_model": NEEDLE_GEN_MODEL,
            "judge_model": JUDGE_MODEL,
            "embed_model": EMBED_MODEL,
            "keyword_match_threshold": KEYWORD_MATCH_THRESHOLD,
            "semantic_sim_threshold": SEMANTIC_SIM_THRESHOLD,
            "needle_fraction_mean": mean_frac,
            "needle_fraction_range": [min_frac, max_frac],
            "niah_compliance": niah_compliant,
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
