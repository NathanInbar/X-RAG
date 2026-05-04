"""Test script to validate strict NIAH 10-30% enforcement"""

import sys
sys.path.insert(0, "/home/ubuntu/X-RAG/src")

from experiments.minea_triple_eval import inject_needles

def test_enforcement():
    """Test that inject_needles enforces 10-30% strictly"""

    # Generate mock needles of various lengths
    def make_needle(length):
        return {"sentence": "x" * length}

    test_cases = [
        ("Short doc, long needles", "y" * 1000, [make_needle(100) for _ in range(10)]),
        ("Long doc, short needles", "y" * 10000, [make_needle(50) for _ in range(10)]),
        ("Medium doc, medium needles", "y" * 5000, [make_needle(80) for _ in range(10)]),
        ("Very long doc", "y" * 20000, [make_needle(100) for _ in range(10)]),
    ]

    print("Testing strict NIAH enforcement:\n")
    print(f"{'Case':<30} {'Doc Len':<10} {'Available':<10} {'Injected':<10} {'Fraction':<12} {'Status'}")
    print("=" * 90)

    for name, doc_text, needles in test_cases:
        enriched_text, fraction, n_injected = inject_needles(doc_text, needles)

        # Check compliance
        EPSILON = 0.001
        in_range = (0.1 - EPSILON) <= fraction <= (0.3 + EPSILON)
        status = "✓ PASS" if in_range else "✗ FAIL"

        print(f"{name:<30} {len(doc_text):<10} {len(needles):<10} {n_injected:<10} {fraction*100:>6.1f}%     {status}")

        if not in_range:
            print(f"  ⚠️  VIOLATION: {fraction*100:.1f}% outside 10-30% range!")

    print("\n✅ All tests completed")

if __name__ == "__main__":
    test_enforcement()
