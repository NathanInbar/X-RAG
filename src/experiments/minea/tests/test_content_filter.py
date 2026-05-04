"""
Test the content validation filter on existing MINEA documents.
Shows which documents would be excluded and which chunks would change.
"""
import re
import json
from pathlib import Path


def is_valid_content_for_extraction(text: str) -> tuple[bool, str]:
    """Content validation filter (same as in minea_triple_eval.py)"""
    if not text or len(text) < 100:
        return False, "too_short"

    text_lower = text.lower()

    # Check 1: ToC structure (short lines with few words)
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if len(lines) > 5:
        avg_line_len = sum(len(l) for l in lines) / len(lines)
        if avg_line_len < 40:
            words_per_line = sum(len(l.split()) for l in lines) / len(lines)
            if words_per_line < 6:
                return False, "toc_structure"

    # Check 2: High citation density
    citation_pattern = re.findall(r'\[\d{1,3}\]', text)
    citation_density = len(citation_pattern) / (len(text) / 100)
    if citation_density > 3:
        return False, "reference_list"

    # Check 3: Explicit ToC keywords
    toc_indicators = ['contents', 'table of contents', 'list of figures', 'list of tables']
    if any(indicator in text_lower[:200] for indicator in toc_indicators):
        return False, "explicit_toc"

    # Check 4: Dot leaders (ToC pagination)
    dot_leader_count = text.count('. . .') + text.count('.....')
    if dot_leader_count > 5:
        return False, "toc_pagination"

    # Check 5: Section numbering index
    section_lines = re.findall(r'^\s*\d+\.\d+.*?\d{1,3}\s*$', text, re.MULTILINE)
    if len(section_lines) > 8:
        return False, "section_index"

    # Check 6: Sufficient narrative sentences
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 30]
    if len(sentences) < 3:
        return False, "insufficient_sentences"

    # Check 7: Has verbs (relationships)
    common_verbs = ['is', 'are', 'was', 'were', 'be', 'been', 'has', 'have', 'had',
                    'can', 'could', 'will', 'would', 'show', 'shows', 'use', 'uses']
    has_verbs = any(f' {verb} ' in text_lower for verb in common_verbs)
    if not has_verbs:
        return False, "no_verbs"

    return True, "valid"


def main():
    # Load current MINEA results to get document list
    with open('/home/ubuntu/X-RAG/src/experiments/minea_results.json') as f:
        data = json.load(f)

    doc_files = [doc['source_file'] for doc in data['results']['opus-4.5']['per_document']]

    CACHE_DIR = Path('/home/ubuntu/X-RAG/src/preprocess_cache')
    MIN_CHUNK_TOKENS = 250

    print("CONTENT VALIDATION ANALYSIS")
    print("=" * 80)
    print("Testing which documents/chunks would change with new filter\n")

    excluded_docs = []
    changed_docs = []
    valid_docs = []

    for doc_file in doc_files:
        cache_file = CACHE_DIR / f"{Path(doc_file).stem}__chunks.json"

        with open(cache_file) as f:
            doc_entries = json.load(f)

        # Get all chunks >= MIN_CHUNK_TOKENS
        all_chunks = []
        for doc in doc_entries:
            for chunk in doc["chunks"]:
                if chunk["approx_n_tokens"] >= MIN_CHUNK_TOKENS:
                    text = chunk["raw_text"]
                    is_val, reason = is_valid_content_for_extraction(text)
                    all_chunks.append({
                        "text": text,
                        "tokens": chunk["approx_n_tokens"],
                        "is_valid": is_val,
                        "reason": reason
                    })

        if not all_chunks:
            continue

        # Current selection (longest)
        current_best = max(all_chunks, key=lambda c: c["tokens"])

        # New selection (valid, then longest)
        valid_chunks = [c for c in all_chunks if c["is_valid"]]

        if not valid_chunks:
            # Document would be EXCLUDED
            excluded_docs.append((doc_file, current_best["reason"]))
            print(f"❌ EXCLUDED: {doc_file[:55]}")
            print(f"   Reason: {current_best['reason']}")
            print(f"   Preview: {current_best['text'][:100]}...\n")
        elif not current_best["is_valid"]:
            # Document chunk would CHANGE
            new_best = max(valid_chunks, key=lambda c: c["tokens"])
            changed_docs.append(doc_file)
            print(f"🔄 CHANGED: {doc_file[:55]}")
            print(f"   Old: {current_best['tokens']} tokens ({current_best['reason']})")
            print(f"   New: {new_best['tokens']} tokens (valid)")
            print(f"   New preview: {new_best['text'][:100]}...\n")
        else:
            # Document stays the SAME
            valid_docs.append(doc_file)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print(f"  ✓ Valid (no change): {len(valid_docs)}")
    print(f"  🔄 Changed chunk:    {len(changed_docs)}")
    print(f"  ❌ Excluded:         {len(excluded_docs)}")
    print(f"  Total:               {len(doc_files)}")

    print(f"\nAfter filtering: {len(valid_docs) + len(changed_docs)}/{len(doc_files)} documents remain")

    if excluded_docs:
        print("\nExcluded documents should be replaced with new samples to reach target size.")


if __name__ == "__main__":
    main()
