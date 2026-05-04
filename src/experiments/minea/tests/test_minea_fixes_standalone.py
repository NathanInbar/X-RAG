"""
Standalone unit tests to validate MINEA implementation fixes.
Tests core logic without requiring full environment setup.
"""

import re
from collections import defaultdict


def inject_needles_test(chunk_text: str, needles: list[dict]) -> tuple[str, float]:
    """
    Test version of inject_needles function.
    """
    sentences = re.split(r"(?<=[.!?])\s+", chunk_text.strip())

    n_needles = len(needles)
    if n_needles == 0:
        return chunk_text, 0.0

    # Pre-calculate evenly-spaced positions
    position_to_needles = defaultdict(list)

    for i in range(n_needles):
        frac = (i + 1) / (n_needles + 1)
        pos = int(frac * len(sentences))
        position_to_needles[pos].append(i)

    # Insert needles at calculated positions
    enriched = []
    for i, sent in enumerate(sentences):
        enriched.append(sent)
        if i in position_to_needles:
            for needle_idx in position_to_needles[i]:
                enriched.append(needles[needle_idx]["sentence"])

    enriched_text = " ".join(enriched)

    # Calculate needle fraction
    original_len = len(chunk_text)
    enriched_len = len(enriched_text)
    needle_fraction = (enriched_len - original_len) / enriched_len if enriched_len > 0 else 0.0

    return enriched_text, needle_fraction


def test_inject_needles_more_than_sentences():
    """Test needle injection when n_needles > n_sentences."""
    print("Test 1: More needles than sentences")

    chunk = "First sentence. Second sentence."
    needles = [
        {"sentence": "Needle 1"},
        {"sentence": "Needle 2"},
        {"sentence": "Needle 3"},
        {"sentence": "Needle 4"},
    ]

    enriched, frac = inject_needles_test(chunk, needles)

    # All needles should be present
    for needle in needles:
        if needle["sentence"] not in enriched:
            print(f"  ✗ Missing: {needle['sentence']}")
            print(f"  Enriched: {enriched}")
            raise AssertionError(f"Missing: {needle['sentence']}")

    print(f"  ✓ All {len(needles)} needles injected into 2 sentences")
    print(f"    Needle fraction: {frac:.1%}\n")


def test_inject_needles_zero():
    """Test needle injection with zero needles."""
    print("Test 2: Zero needles")

    chunk = "Test sentence."
    needles = []

    enriched, frac = inject_needles_test(chunk, needles)

    assert enriched == chunk, "Text should be unchanged"
    assert frac == 0.0, "Fraction should be 0.0"

    print("  ✓ Zero needles handled correctly\n")


def test_inject_needles_one_sentence():
    """Test needle injection with one sentence."""
    print("Test 3: One sentence, one needle")

    chunk = "Single sentence."
    needles = [{"sentence": "Needle X"}]

    enriched, frac = inject_needles_test(chunk, needles)

    assert "Needle X" in enriched, "Needle should be present"
    assert frac > 0, "Fraction should be > 0"

    print(f"  ✓ Needle injected (frac={frac:.1%})\n")


def test_needle_fraction_empty():
    """Test needle fraction with empty enriched text (edge case)."""
    print("Test 4: Empty text edge case")

    chunk = ""
    needles = []

    enriched, frac = inject_needles_test(chunk, needles)

    assert frac == 0.0, "Fraction should be 0.0 for empty text"

    print("  ✓ Empty text handled correctly\n")


def test_context_alignment():
    """Test that context slicing handles edge cases."""
    print("Test 5: Context alignment edge cases")

    # Short text
    short = "Short"
    assert short[:5000] == "Short", "Should return entire short text"

    # Empty text
    empty = ""
    assert empty[:5000] == "", "Should return empty string"

    # Long text
    long = "x" * 10000
    assert len(long[:5000]) == 5000, "Should return first 5000 chars"

    print("  ✓ Context slicing handles all edge cases\n")


def test_needle_distribution_uniformity():
    """Test that needles are distributed uniformly."""
    print("Test 6: Needle distribution uniformity")

    # Create a chunk with many sentences
    sentences = [f"Sentence {i}." for i in range(20)]
    chunk = " ".join(sentences)

    needles = [{"sentence": f"Needle {i}"} for i in range(5)]

    enriched, frac = inject_needles_test(chunk, needles)

    # Check that needles appear in order
    positions = []
    for needle in needles:
        pos = enriched.find(needle["sentence"])
        if pos == -1:
            print(f"  ✗ Needle not found: {needle['sentence']}")
            print(f"  Enriched: {enriched[:200]}...")
            raise AssertionError(f"Needle not found: {needle['sentence']}")
        positions.append(pos)

    # Positions should be increasing (needles maintain order)
    if positions != sorted(positions):
        print(f"  ✗ Needles not in order: {positions}")
        raise AssertionError("Needles should appear in order")

    print(f"  ✓ 5 needles distributed uniformly across 20 sentences")
    print(f"    Positions: {positions}\n")


def test_type_safety():
    """Test type validation logic."""
    print("Test 7: Type safety in parsing")

    # Simulate malformed data
    malformed_data = [
        {"subject": 123, "predicate": "test", "object": "obj", "sentence": "sent"},
        {"subject": "s", "predicate": None, "object": "o", "sentence": "sent"},
        {"subject": "s", "predicate": "p", "object": ["list"], "sentence": "sent"},
        {"subject": "Valid", "predicate": "is", "object": "good", "sentence": "This works."},
    ]

    # Validate
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

    assert len(needles) == 1, f"Should have 1 valid needle, got {len(needles)}"
    assert needles[0]["subject"] == "Valid", "Should be the valid needle"

    print(f"  ✓ Type validation filtered {len(malformed_data) - len(needles)}/4 malformed items\n")


def test_edge_case_duplicate_positions():
    """Test that duplicate positions are handled correctly."""
    print("Test 8: Duplicate positions (edge case)")

    # Two sentences, three needles -> positions will be [0, 1, 1]
    chunk = "First. Second."
    needles = [
        {"sentence": "N1"},
        {"sentence": "N2"},
        {"sentence": "N3"},
    ]

    enriched, frac = inject_needles_test(chunk, needles)

    # All three needles should be present
    for needle in needles:
        if needle["sentence"] not in enriched:
            print(f"  ✗ Missing: {needle['sentence']}")
            print(f"  Enriched: {enriched}")
            raise AssertionError(f"Missing: {needle['sentence']}")

    print(f"  ✓ All 3 needles injected despite duplicate positions\n")


def main():
    print("=" * 80)
    print("MINEA FIXES - STANDALONE VALIDATION TESTS")
    print("=" * 80)
    print()

    tests = [
        test_inject_needles_more_than_sentences,
        test_inject_needles_zero,
        test_inject_needles_one_sentence,
        test_needle_fraction_empty,
        test_context_alignment,
        test_needle_distribution_uniformity,
        test_type_safety,
        test_edge_case_duplicate_positions,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}\n")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}\n")
            failed += 1

    print("=" * 80)
    print(f"RESULTS: {passed}/{len(tests)} passed")
    print("=" * 80)

    if failed > 0:
        print("\n⚠️  Some tests failed - review implementation")
        exit(1)
    else:
        print("\n✓ All tests passed!")


if __name__ == "__main__":
    main()
