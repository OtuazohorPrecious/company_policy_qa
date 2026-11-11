from app.checker import allowed_numbers, check_sentences, number_guard, split_sentences


def by_id(sections):
    return {s.chunk_id: s for s in sections}


def pick(sections, *ids):
    lookup = by_id(sections)
    return [lookup[i] for i in ids]


def test_split_sentences_keeps_amounts_together():
    text = "You owe NGN 150,000. Unused leave is paid out at 1.5 times? No. It is paid at the daily rate."
    assert split_sentences(text) == ["You owe NGN 150,000.", "Unused leave is paid out at 1.5 times?",
                                     "No.", "It is paid at the daily rate."]


def test_number_guard_allows_percentage_calculation(sections):
    q12 = ("The company paid NGN 300,000 for my certification. If I resign 8 months after finishing it, "
           "what do I owe, and what happens to my unused annual leave?")
    secs = pick(sections, "07_learning_and_development_policy#4", "01_annual_leave_policy#6")
    allowed = allowed_numbers(q12, secs)
    assert number_guard("You need to repay NGN 150,000.", allowed) == []      # 50% x 300,000
    assert number_guard("You need to repay NGN 175,000.", allowed) == [175000.0]


def test_number_guard_allows_question_and_section_numbers(sections):
    q = "I stayed in a hotel costing NGN 70,000 per night. Is that within the limit?"
    allowed = allowed_numbers(q, pick(sections, "04_business_travel_policy#4"))
    assert number_guard("NGN 70,000 is within the NGN 80,000 limit.", allowed) == []
    assert number_guard("The limit is NGN 90,000.", allowed) == [90000.0]


def test_guard_lowers_a_supported_verdict(sections):
    secs = pick(sections, "04_business_travel_policy#4")
    labels = [{"index": 0, "verdict": "supported", "doc_id": "04_business_travel_policy", "section": "4",
               "evidence": "Domestic hotels: maximum NGN 80,000 per night."}]
    [result] = check_sentences(["The domestic hotel limit is NGN 95,000 per night."], labels, "hotel limit?",
                               secs, by_id(sections))
    assert result.verdict == "not_supported" and "number guard" in result.reason


def test_missing_label_counts_as_failure(sections):
    secs = pick(sections, "04_business_travel_policy#4")
    [result] = check_sentences(["Some sentence."], [], "q", secs, by_id(sections))
    assert not result.passed and result.reason == "checker returned no label"
