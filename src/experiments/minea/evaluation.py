"""
Needle extraction evaluation criteria for MINEA.

Implements multiple identification methods:
- Exact match (string equality)
- Keyword overlap (term presence)
- Semantic similarity (embedding-based)
- LLM-as-judge (contextual matching)
"""

import asyncio
import json
import logging
import re
from typing import Any

import dspy
import litellm
import numpy as np

logger = logging.getLogger(__name__)


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
    needle: dict, extracted: list[dict], threshold: float
) -> bool:
    """
    Check if needle keywords appear in extracted triples.

    A needle is matched if at least `threshold` fraction of its keywords
    appear in any extracted triple's S/P/O fields (word-boundary match).
    """
    keywords = needle.get("keywords", [])
    if not keywords:
        return False

    # Pre-compile regex patterns for efficiency
    kw_patterns = [
        re.compile(r"\b" + re.escape(kw.lower().strip()) + r"\b")
        for kw in keywords
    ]

    for t in extracted:
        triple_text = f"{t['s']} {t['p']} {t['o']}".lower()
        matched = sum(1 for pattern in kw_patterns if pattern.search(triple_text))
        if matched / len(kw_patterns) >= threshold:
            return True

    return False


async def _embed_texts(
    texts: list[str],
    embed_model: str,
    embed_batch_limit: int,
    initial_delay: float,
    max_attempts: int,
) -> list[np.ndarray]:
    """Embed texts via embedding model (batched)."""
    if not texts:
        return []

    all_embeddings: list[np.ndarray] = []
    for i in range(0, len(texts), embed_batch_limit):
        batch = texts[i : i + embed_batch_limit]
        delay = initial_delay
        for attempt in range(1, max_attempts + 1):
            try:
                resp = await litellm.aembedding(model=embed_model, input=batch)
                sorted_data = sorted(resp.data, key=lambda d: d["index"])
                all_embeddings.extend(np.array(d["embedding"]) for d in sorted_data)
                break
            except Exception as e:
                if attempt < max_attempts:
                    logger.warning(
                        f"[embed] attempt {attempt} failed: {e}, retrying in {delay}s"
                    )
                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    raise
    return all_embeddings


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two embedding vectors."""
    d = np.dot(a, b)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(d / n) if n > 0 else 0.0


async def semantic_match(
    needle: dict,
    extracted_embeddings: list[np.ndarray],
    embed_model: str,
    embed_batch_limit: int,
    initial_delay: float,
    max_attempts: int,
    threshold: float,
) -> bool:
    """
    Check if needle is semantically similar to any extracted triple.

    Args:
        needle: Needle dict with sentence/subject/predicate/object
        extracted_embeddings: Pre-computed embeddings for extracted triples
        embed_model: Embedding model ID
        embed_batch_limit: Max embeddings per batch
        initial_delay: Initial retry delay
        max_attempts: Max retry attempts
        threshold: Cosine similarity threshold

    Returns:
        True if max similarity exceeds threshold
    """
    if not extracted_embeddings:
        return False

    needle_sent = needle.get("sentence", "")
    if not needle_sent:
        needle_sent = f"{needle['subject']} {needle['predicate']} {needle['object']}"

    try:
        needle_emb = (await _embed_texts(
            [needle_sent], embed_model, embed_batch_limit, initial_delay, max_attempts
        ))[0]
        max_sim = max(_cosine_sim(needle_emb, te) for te in extracted_embeddings)
        return max_sim >= threshold

    except Exception as e:
        logger.error(f"[semantic_match] error: {e}")
        return False


async def llm_judge_match(
    needle: dict,
    extracted: list[dict],
    judge_model: str,
    initial_delay: float,
    max_attempts: int,
) -> bool:
    """Use LLM judge to determine if needle was captured."""
    if not extracted:
        return False

    lm = dspy.LM(judge_model)
    predict = dspy.Predict(_JudgeNeedleMatch)

    extracted_for_judge = [
        {"subject": t["s"], "predicate": t["p"], "object": t["o"]} for t in extracted
    ]
    extracted_json = json.dumps(extracted_for_judge, ensure_ascii=False)

    delay = initial_delay
    for attempt in range(1, max_attempts + 1):
        try:
            with dspy.context(lm=lm):
                pred = await predict.acall(
                    needle_subject=needle["subject"],
                    needle_predicate=needle["predicate"],
                    needle_object=needle["object"],
                    extracted_triples_json=extracted_json,
                )

            is_captured = getattr(pred, "is_captured", False)
            # Robust parsing - convert to string then check
            return str(is_captured).strip().lower() in ("true", "yes", "1")

        except Exception as e:
            if attempt < max_attempts:
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
    needles: list[dict],
    extraction_result: dict,
    embed_model: str,
    judge_model: str,
    embed_batch_limit: int,
    initial_delay: float,
    max_attempts: int,
    keyword_match_threshold: float,
    semantic_sim_threshold: float,
    max_concurrent_judge_calls: int,
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

    # Pre-compute extracted triple embeddings (shared across all needles)
    extracted_sents = [f"{t['s']} {t['p']} {t['o']}" for t in extracted]
    extracted_embeddings = (
        await _embed_texts(
            extracted_sents, embed_model, embed_batch_limit, initial_delay, max_attempts
        )
        if extracted
        else []
    )

    per_needle = []

    # Synchronous checks (exact, keyword)
    for needle in needles:
        exact = exact_match(needle, extracted)
        keyword = keyword_match(needle, extracted, keyword_match_threshold)
        per_needle.append({
            "needle": needle,
            "exact_match": exact,
            "keyword_match": keyword,
            "semantic_match": None,  # will fill async
            "judge_match": None,  # will fill async
        })

    # Parallel semantic matching (using pre-computed embeddings)
    sem_tasks = [
        semantic_match(
            needle,
            extracted_embeddings,
            embed_model,
            embed_batch_limit,
            initial_delay,
            max_attempts,
            semantic_sim_threshold,
        )
        for needle in needles
    ]
    sem_results = await asyncio.gather(*sem_tasks)
    for i, sem in enumerate(sem_results):
        per_needle[i]["semantic_match"] = sem

    # Parallel judge matching with concurrency limit to avoid rate limits
    sem = asyncio.Semaphore(max_concurrent_judge_calls)

    async def judge_with_limit(needle):
        async with sem:
            return await llm_judge_match(needle, extracted, judge_model, initial_delay, max_attempts)

    judge_tasks = [judge_with_limit(needle) for needle in needles]
    judge_results = await asyncio.gather(*judge_tasks)
    for i, judge in enumerate(judge_results):
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
