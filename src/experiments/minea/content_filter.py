"""
Content quality filtering for MINEA evaluation.

Rejects structural/metadata sections that lack extractable relationships:
- Table of contents (section headers + page numbers)
- Reference/bibliography lists (citations without narrative)
- Author lists / affiliations
- Pure figure/table captions
"""

import re


def is_valid_content_for_extraction(
    text: str,
    min_content_length: int,
    max_avg_line_len_toc: int,
    citation_density_threshold: float,
) -> tuple[bool, str]:
    """
    Determine if a chunk contains valid content for triple extraction evaluation.

    Returns (is_valid, reason) where reason explains why content was rejected.

    Rejects structural/metadata sections that lack extractable relationships:
    - Table of contents (section headers + page numbers)
    - Reference/bibliography lists (citations without narrative)
    - Author lists / affiliations
    - Pure figure/table captions

    This is NOT a quality filter - we don't judge if content is "good academic writing."
    We only filter sections that fundamentally lack subject-predicate-object relationships.
    """
    if not text or len(text) < min_content_length:
        return False, "too_short"

    text_lower = text.lower()

    # Check 1: Excessive line breaks with short content (ToC/list structure)
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if len(lines) > 5:
        avg_line_len = sum(len(l) for l in lines) / len(lines)
        if avg_line_len < max_avg_line_len_toc:
            # Very short lines suggest headers/ToC rather than narrative
            # But check if it's just a paragraph with many newlines
            words_per_line = sum(len(l.split()) for l in lines) / len(lines)
            if words_per_line < 6:
                return False, "toc_structure"

    # Check 2: High citation density (reference list)
    # Pattern: [1], [23], [145] etc.
    citation_pattern = re.findall(r'\[\d{1,3}\]', text)
    citation_density = len(citation_pattern) / (len(text) / 100)  # citations per 100 chars
    if citation_density > citation_density_threshold:
        return False, "reference_list"

    # Check 3: Table of contents keywords
    toc_indicators = ['contents', 'table of contents', 'list of figures', 'list of tables']
    if any(indicator in text_lower[:200] for indicator in toc_indicators):
        return False, "explicit_toc"

    # Check 4: Excessive dot leaders (ToC pagination)
    # Pattern: ". . . . ." or "....."
    dot_leader_count = text.count('. . .') + text.count('.....')
    if dot_leader_count > 5:
        return False, "toc_pagination"

    # Check 5: Section numbering without narrative
    # Pattern: Multiple lines starting with "X.Y SomethingTitle" followed by numbers (page refs)
    section_lines = re.findall(r'^\s*\d+\.\d+.*?\d{1,3}\s*$', text, re.MULTILINE)
    if len(section_lines) > 8:
        return False, "section_index"

    # Check 6: Verify presence of narrative sentences with verbs
    # (Distinguishes content from pure lists/headers)
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 30]
    if len(sentences) < 3:
        return False, "insufficient_sentences"

    # Basic verb check - academic content has verbs describing relationships
    common_verbs = ['is', 'are', 'was', 'were', 'be', 'been', 'has', 'have', 'had',
                    'can', 'could', 'will', 'would', 'show', 'shows', 'use', 'uses']
    has_verbs = any(f' {verb} ' in text_lower for verb in common_verbs)
    if not has_verbs:
        return False, "no_verbs"

    return True, "valid"
