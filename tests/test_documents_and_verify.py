from app.verify import normalise, split_citations, verify_citation


def test_one_chunk_per_section(sections):
    assert len(sections) == 48
    ids = {s.chunk_id for s in sections}
    assert "04_business_travel_policy#4" in ids
    travel4 = next(s for s in sections if s.chunk_id == "04_business_travel_policy#4")
    assert travel4.title == "Business Travel Policy"
    assert travel4.heading == "4. Accommodation"
    assert travel4.index_text.startswith("Business Travel Policy · 4. Accommodation")
    assert "FICTIONAL" not in travel4.body


def test_normalise_ignores_markdown_and_spaces():
    assert normalise("- Domestic hotels: maximum **NGN 80,000   per night**.") == \
        "Domestic hotels: maximum NGN 80,000 per night."


def by_id(sections):
    return {s.chunk_id: s for s in sections}


def test_exact_quote_verifies_and_returns_source_text(sections):
    c = verify_citation(
        {"doc_id": "04_business_travel_policy", "section": "4",
         "quote": "Domestic hotels: maximum NGN 80,000 per night."},
        by_id(sections),
    )
    assert c is not None
    assert c.quote == "Domestic hotels: maximum **NGN 80,000 per night**."


def test_paraphrase_wrong_section_and_unknown_doc_fail(sections):
    lookup = by_id(sections)
    assert verify_citation({"doc_id": "04_business_travel_policy", "section": "4",
                            "quote": "Domestic hotels cost at most NGN 80,000 a night."}, lookup) is None
    assert verify_citation({"doc_id": "04_business_travel_policy", "section": "5",
                            "quote": "Domestic hotels: maximum NGN 80,000 per night."}, lookup) is None
    assert verify_citation({"doc_id": "99_missing", "section": "1", "quote": "anything"}, lookup) is None
    assert verify_citation({"doc_id": "04_business_travel_policy", "section": "4", "quote": "  ** "}, lookup) is None


def test_doc_id_with_md_suffix_is_accepted(sections):
    c = verify_citation({"doc_id": "02_sick_leave_policy.md", "section": "3",
                         "quote": "more than 2 consecutive working days"}, by_id(sections))
    assert c is not None and c.doc_id == "02_sick_leave_policy"


def test_split_citations_removes_duplicates(sections):
    good = {"doc_id": "02_sick_leave_policy", "section": "1", "quote": "10 paid sick days"}
    bad = {"doc_id": "02_sick_leave_policy", "section": "1", "quote": "twelve sick days"}
    verified, invalid = split_citations([good, good, bad], by_id(sections))
    assert len(verified) == 1 and invalid == [bad]

