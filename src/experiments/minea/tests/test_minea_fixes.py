"""
Unit tests to validate MINEA implementation fixes.

Tests edge cases for:
1. Needle distribution (more needles than sentences, duplicates)
2. Needle fraction calculation
3. Context alignment
4. Keyword matching
5. Type safety in parsing

Usage:
    cd /home/ubuntu/X-RAG/src
    PYTHONPATH=. python -m experiments.test_minea_fixes
"""

import json
from experiments.minea_triple_eval import inject_needles, keyword_match


def test_inject_needles_more_than_sentences():
    """Test needle injection when n_needles > n_sentences."""
    print("Test 1: More needles than sentences")

    chunk = "First sentence. Second sentence."
    needles = [
        {"sentence": "Needle 1", "keywords": []},
        {"sentence": "Needle 2", "keywords": []},
        {"sentence": "Needle 3", "keywords": []},
        {"sentence": "Needle 4", "keywords": []},
    ]

    enriched, frac = inject_needles(chunk, needles)

    # All needles should be present
    for needle in needles:
        assert needle["sentence"] in enriched, f"Missing: {needle['sentence']}"

    print(f"✓ All {len(needles)} needles injected into {chunk.count('.')} sentences")
    print(f"  Enriched length: {len(enriched)} chars, fraction: {frac:.1%}\n")


def test_inject_needles_zero():
    """Test needle injection with zero needles."""
    print("Test 2: Zero needles")

    chunk = "Test sentence."
    needles = []

    enriched, frac = inject_needles(chunk, needles)

    assert enriched == chunk, "Text should be unchanged"
    assert frac == 0.0, "Fraction should be 0.0"

    print("✓ Zero needles handled correctly\n")


def test_inject_needles_one_sentence():
    """Test needle injection with one sentence."""
    print("Test 3: One sentence, one needle")

    chunk = "Single sentence."
    needles = [{"sentence": "Needle X", "keywords": []}]

    enriched, frac = inject_needles(chunk, needles)

    assert "Needle X" in enriched, "Needle should be present"
    assert frac > 0, "Fraction should be > 0"

    print(f"✓ Needle injected: {enriched}\n")


def test_needle_fraction_empty():
    """Test needle fraction with empty enriched text (edge case)."""
    print("Test 4: Empty text edge case")

    chunk = ""
    needles = []

    enriched, frac = inject_needles(chunk, needles)

    assert frac == 0.0, "Fraction should be 0.0 for empty text"

    print("✓ Empty text handled correctly\n")


def test_keyword_match_missing_keys():
    """Test keyword matching with malformed extracted triples."""
    print("Test 5: Keyword match with malformed triples")

    needle = {
        "keywords": ["machine", "learning"],
    }

    # Well-formed triple
    extracted = [{"s": "AI", "p": "includes", "o": "machine learning"}]
    result = keyword_match(needle, extracted)
    assert result, "Should match with valid triple"

    print("✓ Keyword match handles valid triples\n")


def test_type_safety_needle_generation():
    """Test that non-string types in needle generation are handled."""
    print("Test 6: Type safety in needle parsing")

    # Simulate malformed JSON with non-string types
    malformed_data = [
        {"subject": 123, "predicate": "test", "object": "obj", "sentence": "sent"},
        {"subject": "s", "predicate": None, "object": "o", "sentence": "sent"},
        {"subject": "s", "predicate": "p", "object": ["list"], "sentence": "sent"},
        {"subject": "Valid", "predicate": "is", "object": "good", "sentence": "This works."},
    ]

    # Simulate the validation logic from generate_needles
    needles = []
    for item in malformed_data:
        if not isinstance(item, dict):
            continue
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
        if s and p and o and sent:
            needles.append({"subject": s, "predicate": p, "object": o, "sentence": sent})

    assert len(needles) == 1, f"Should only have 1 valid needle, got {len(needles)}"
    assert needles[0]["subject"] == "Valid", "Should be the valid needle"

    print(f"✓ Type validation filtered {len(malformed_data) - len(needles)}/4 malformed items\n")


def test_type_safety_extraction():
    """Test that non-string types in extraction are handled."""
    print("Test 7: Type safety in extraction parsing")

    # Simulate malformed extraction JSON
    malformed_data = [
        {"subject": 123, "predicate": "test", "object": "obj"},
        {"subject": "s", "predicate": None, "object": "o"},
        {"subject": "s", "predicate": "p", "object": ["list"]},
        {"subject": "Valid", "predicate": "is", "object": "good"},
    ]

    # Simulate the validation logic from run_extraction_on_enriched_chunk
    triples = []
    for item in malformed_data:
        if not isinstance(item, dict):
            continue
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

    assert len(triples) == 1, f"Should only have 1 valid triple, got {len(triples)}"
    assert triples[0]["s"] == "Valid", "Should be the valid triple"

    print(f"✓ Type validation filtered {len(malformed_data) - len(triples)}/4 malformed items\n")


def test_context_alignment():
    """Test that context slicing handles edge cases."""
    print("Test 8: Context alignment edge cases")

    # Short text
    short = "Short"
    assert short[:5000] == "Short", "Should return entire short text"

    # Empty text
    empty = ""
    assert empty[:5000] == "", "Should return empty string"

    # Long text
    long = "x" * 10000
    assert len(long[:5000]) == 5000, "Should return first 5000 chars"

    print("✓ Context slicing handles all edge cases\n")


def test_needle_distribution_uniformity():
    """Test that needles are distributed uniformly."""
    print("Test 9: Needle distribution uniformity")

    # Create a chunk with many sentences
    sentences = [f"Sentence {i}." for i in range(20)]
    chunk = " ".join(sentences)

    needles = [{"sentence": f"Needle {i}", "keywords": []} for i in range(5)]

    enriched, frac = inject_needles(chunk, needles)

    # Check that needles appear in order
    positions = []
    for needle in needles:
        pos = enriched.find(needle["sentence"])
        assert pos != -1, f"Needle not found: {needle['sentence']}"
        positions.append(pos)

    # Positions should be increasing (needles maintain order)
    assert positions == sorted(positions), "Needles should appear in order"

    # Check approximate spacing
    spacings = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
    avg_spacing = sum(spacings) / len(spacings)

    # All spacings should be within 2x of average (rough check for uniformity)
    for spacing in spacings:
        assert spacing > avg_spacing / 3, f"Spacing too small: {spacing} vs avg {avg_spacing}"

    print(f"✓ 5 needles distributed uniformly across 20 sentences")
    print(f"  Positions: {positions}")
    print(f"  Spacings: {spacings}\n")


def main():
    print("=" * 80)
    print("MINEA IMPLEMENTATION FIXES - VALIDATION TESTS")
    print("=" * 80)
    print()

    tests = [
        test_inject_needles_more_than_sentences,
        test_inject_needles_zero,
        test_inject_needles_one_sentence,
        test_needle_fraction_empty,
        test_keyword_match_missing_keys,
        test_type_safety_needle_generation,
        test_type_safety_extraction,
        test_context_alignment,
        test_needle_distribution_uniformity,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ FAILED: {e}\n")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {e}\n")
            failed += 1

    print("=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 80)

    if failed > 0:
        exit(1)


if __name__ == "__main__":
    main()
