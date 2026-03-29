#!/usr/bin/env python3
"""
Convert ULTRA dataset JSONL files to our internal format.

Takes the contexts from the UltraDomain dataset and pre-processes them into
ingestable articles. Groups multiple questions by their shared context.
"""

import json
from pathlib import Path
from collections import defaultdict
from xrag.paths import DATASETS_DIR

def convert_ultra_file(input_path: Path, output_dir: Path):
    """Convert a single ULTRA JSONL file to our format."""

    # Group entries by context (each context is a separate document)
    context_groups = defaultdict(lambda: {"queries": [], "answers": []})

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())

            context = entry["context"]
            question = entry["input"]
            answers = entry["answers"]  # This is already a list

            # Add this question and its answers to the context group
            context_groups[context]["queries"].append(question)
            context_groups[context]["answers"].append(answers)

    # Write one file per unique context (document), matching eval loop expectations
    domain_name = input_path.stem  # e.g., "agriculture" from "agriculture.jsonl"

    print(f"Processing {input_path.name}:")
    print(f"  Found {len(context_groups)} unique contexts")

    for idx, (context, data) in enumerate(context_groups.items(), start=1):
        output_entry = {
            "essay": context,
            "queries": data["queries"],
            "answers": data["answers"]
        }
        output_path = output_dir / f"{domain_name}_{idx:04d}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_entry, f, indent=2, ensure_ascii=False)

    print(f"  Written {len(context_groups)} files to {output_dir}")
    return len(context_groups)

def main():
    """Convert all ULTRA dataset files."""

    input_dir = DATASETS_DIR / "ULTRA" / "original"
    output_dir = DATASETS_DIR / "ULTRA"

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # Find all JSONL files in the input directory
    jsonl_files = list(input_dir.glob("*.jsonl"))

    if not jsonl_files:
        print(f"No JSONL files found in {input_dir}")
        return

    total_contexts = 0
    for jsonl_file in sorted(jsonl_files):
        num_contexts = convert_ultra_file(jsonl_file, output_dir)
        total_contexts += num_contexts
        print()

    print(f"Conversion complete! Total unique contexts across all files: {total_contexts}")

if __name__ == "__main__":
    main()
