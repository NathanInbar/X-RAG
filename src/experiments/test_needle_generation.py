"""
Quick test script to validate needle generation and injection.

This script generates needles for a single document and shows
what the enriched text looks like, without running full extraction.

Usage:
    cd /home/ubuntu/X-RAG/src
    PYTHONPATH=. python -m experiments.test_needle_generation
"""

import asyncio
import json
import random
from pathlib import Path

import dspy
from dspy.adapters.chat_adapter import ChatAdapter

from experiments.minea_triple_eval import (
    generate_needles,
    inject_needles,
    NEEDLE_GEN_MODEL,
    DATASETS_DIR,
    DATASET,
)


async def main():
    dspy.configure(adapter=ChatAdapter())

    # load one random document
    dataset_dir = DATASETS_DIR / DATASET
    all_files = sorted(dataset_dir.glob("*.json"))
    if not all_files:
        print(f"No files found in {dataset_dir}")
        return

    sample_file = random.choice(all_files)
    print(f"📄 Loading: {sample_file.name}\n")

    with open(sample_file) as fp:
        data = json.load(fp)

    essay = data.get("essay", "")
    if not essay:
        print("❌ No essay content found")
        return

    # take first 2000 chars
    chunk_text = essay[:2000]
    print("=" * 80)
    print("ORIGINAL TEXT (first 2000 chars)")
    print("=" * 80)
    print(chunk_text)
    print()

    # generate needles
    print("=" * 80)
    print("GENERATING NEEDLES...")
    print("=" * 80)
    try:
        needles = await generate_needles(chunk_text, n_needles=3)
        print(f"✅ Generated {len(needles)} needles\n")

        for i, needle in enumerate(needles, 1):
            print(f"Needle {i}:")
            print(f"  Subject:   {needle['subject']}")
            print(f"  Predicate: {needle['predicate']}")
            print(f"  Object:    {needle['object']}")
            print(f"  Keywords:  {', '.join(needle['keywords'])}")
            print(f"  Sentence:  {needle['sentence']}")
            print()

    except Exception as e:
        print(f"❌ Needle generation failed: {e}")
        return

    # inject needles
    print("=" * 80)
    print("INJECTING NEEDLES INTO TEXT...")
    print("=" * 80)
    enriched = inject_needles(chunk_text, needles)

    # highlight injected sentences
    print("ENRICHED TEXT (needle sentences in [NEEDLE]):\n")
    for needle in needles:
        enriched = enriched.replace(
            needle["sentence"], f"[NEEDLE] {needle['sentence']} [/NEEDLE]"
        )

    print(enriched)
    print()

    # stats
    original_len = len(chunk_text)
    enriched_len = len(enriched)
    needle_fraction = (enriched_len - original_len) / enriched_len
    print("=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print(f"Original text length:  {original_len} chars")
    print(f"Enriched text length:  {enriched_len} chars")
    print(f"Needle fraction:       {needle_fraction:.1%}")
    print(f"Target range (NIAH):   10-30%")
    print(
        f"Status:                {'✅ Within range' if 0.1 <= needle_fraction <= 0.3 else '⚠️  Outside recommended range'}"
    )


if __name__ == "__main__":
    asyncio.run(main())
