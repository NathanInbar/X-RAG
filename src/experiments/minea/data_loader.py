"""
Data loading and preprocessing for MINEA evaluation.

Handles loading chunks from preprocessed cache with content quality filtering.
"""

import json
import logging
import random
from pathlib import Path
from typing import Any

from xrag.dataset_processing.preprocess import process_dataset_file
from .content_filter import is_valid_content_for_extraction

logger = logging.getLogger(__name__)


async def load_sample_documents(
    datasets_dir: Path,
    cache_dir: Path,
    dataset: str,
    sample_size: int,
    min_chunk_tokens: int,
    seed: int,
    min_content_length: int,
    max_avg_line_len_toc: int,
    citation_density_threshold: float,
) -> list[dict[str, Any]]:
    """
    Load chunks using SaT segmentation from preprocess cache.

    Uses the same chunking approach as triplet_model_comparison experiment
    to ensure consistency across evaluations.

    Filters chunks by content quality to avoid low-value content like
    tables of contents, reference lists, or header-only sections.
    """
    dataset_dir = datasets_dir / dataset
    all_files = sorted(dataset_dir.glob("*.json"))
    if not all_files:
        raise RuntimeError(f"No JSON files found in {dataset_dir}")

    rng = random.Random(seed)
    sampled = rng.sample(all_files, min(sample_size, len(all_files)))

    # Ensure cache exists for every sampled document (generates if missing)
    for f in sampled:
        cache_file = cache_dir / f"{f.stem}__chunks.json"
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
        cache_file = cache_dir / f"{f.stem}__chunks.json"
        if not cache_file.is_file():
            logger.warning(f"No cache file for {f.name}, skipping")
            continue

        try:
            with open(cache_file) as fp:
                doc_entries = json.load(fp)
        except json.JSONDecodeError as e:
            logger.error(f"Corrupted cache file {cache_file}: {e} — skipping")
            continue

        for doc in doc_entries:
            for chunk in doc["chunks"]:
                if chunk["approx_n_tokens"] < min_chunk_tokens:
                    continue

                text = chunk["raw_text"]
                is_valid, reason = is_valid_content_for_extraction(
                    text, min_content_length, max_avg_line_len_toc, citation_density_threshold
                )

                all_chunks.append({
                    "source_file": f.name,
                    "text": text,
                    "approx_n_tokens": chunk["approx_n_tokens"],
                    "chunk_id": chunk["id"],
                    "is_valid": is_valid,
                    "reject_reason": reason if not is_valid else None,
                })

    # Group chunks by document and select one representative chunk per document
    # This ensures diverse content across the N sampled documents
    doc_to_chunks = {}
    for chunk in all_chunks:
        source = chunk["source_file"]
        if source not in doc_to_chunks:
            doc_to_chunks[source] = []
        doc_to_chunks[source].append(chunk)

    # Select best chunk: filter invalid content, then pick longest
    # Invalid content = ToC, reference lists, structural sections without extractable facts
    # Documents with no valid chunks are EXCLUDED from evaluation
    documents = []
    skipped_docs = []

    for source_file, chunks in doc_to_chunks.items():
        # Filter to valid chunks only
        valid_chunks = [c for c in chunks if c["is_valid"]]

        if not valid_chunks:
            # No valid chunks; skip document entirely
            best_chunk = max(chunks, key=lambda c: c["approx_n_tokens"])
            reason = best_chunk["reject_reason"]
            skipped_docs.append((source_file, reason))
            logger.warning(
                f"{source_file[:40]}: EXCLUDED - no valid chunks "
                f"(longest chunk rejected for: {reason})"
            )
            continue

        # Among valid chunks, pick longest
        best_chunk = max(valid_chunks, key=lambda c: c["approx_n_tokens"])

        documents.append({
            "source_file": source_file,
            "text": best_chunk["text"],
            "approx_n_tokens": best_chunk["approx_n_tokens"],
            "chunk_id": best_chunk["chunk_id"],
        })

    # Log filtering statistics
    logger.info(
        f"Content filtering: {len(documents)} valid documents, "
        f"{len(skipped_docs)} excluded"
    )
    if skipped_docs:
        reason_counts = {}
        for _, reason in skipped_docs:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        for reason, count in reason_counts.items():
            logger.info(f"  Excluded ({reason}): {count} documents")

    # If we excluded documents, sample more to reach target size
    if len(documents) < sample_size and len(skipped_docs) > 0:
        logger.warning(
            f"Only {len(documents)}/{sample_size} valid documents. "
            f"Consider increasing SAMPLE_SIZE or reviewing dataset quality."
        )

    logger.info(f"Loaded {len(documents)} chunks from {len(sampled)} documents (using SaT segmentation)")
    return documents
