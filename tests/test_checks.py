import pytest

from app.checks import check_numbers, quantities, sentences_with_citations


def texts(text):
    return [q.text for q in quantities(text)]


# ---- Which numbers are checked ----


def test_amounts_rates_and_ratios_are_checked():
    assert texts("Revenue was $9.5B, up +24% y/y, margin 24.0%, leverage 2.5x and 1,350 units.") == [
        "$9.5B",
        "+24%",
        "24.0%",
        "2.5x",
        "1,350",
    ]


def test_currency_words_and_magnitudes_are_part_of_the_number():
    assert texts("Profit of AED 22.7 billion and costs of (420)bps; dividend 7.0p.") == [
        "AED 22.7 billion",
        "(420)bps",
        "7.0p",
    ]


@pytest.mark.parametrize(
    "text",
    [
        "In 2025 revenue grew.",  # a year
        "In 4Q'25 and Q3 and H1 and FY25 and 2026E.",  # period labels
        "It has 3 segments.",  # a small count
        "See source 1 and [2].",
    ],
)
def test_years_labels_and_small_counts_are_skipped(text):
    assert texts(text) == []


def test_large_plain_numbers_are_checked():
    assert texts("Headcount reached 69465 people.") == ["69465"]


# ---- Which sources a sentence cites ----


def test_each_sentence_keeps_its_own_citations():
    pairs = sentences_with_citations("Revenue rose 12% [1]. Margin was 18.4% [2][3].", 3)

    assert [cited for _, cited in pairs] == [[1], [2, 3]]
    assert "[1]" not in pairs[0][0]


def test_a_citation_on_its_own_line_covers_the_sentences_before_it():
    answer = "Revenue grew by $9.5B in 4Q'25.\nIt was up +24%.\n\n[1]"

    pairs = sentences_with_citations(answer, 2)

    assert [cited for _, cited in pairs] == [[1], [1]]


def test_citations_to_sources_that_do_not_exist_are_ignored():
    assert sentences_with_citations("Up 5% [9].", 2)[0][1] == []


# ---- Results ----

GE_CHART = (
    "Type: chart\nTitle: Commercial Engines & Services (CES): 4Q'25 performance\n"
    "| Revenue | $7.7B | $8.9B | $9.5B |\n| | +24% y/y |\n"
    "| Operating Profit Margin | 28.2% | 27.4% | 24.0% |\n| | +5% y/y (420)bps y/y |"
)
GE_FULL_YEAR = "|**Commercial Engines & Services (CES)**|**$8.9**|**$(0.3)**|**$8.6**|"


def statuses(result):
    return {n["text"]: (n["status"], n["found_in"]) for n in result["numbers"]}


def test_real_mix_up_from_the_ge_deck_is_caught():
    """Qwen's answer from the Phase 10 check: full-year profit cited to a quarterly chart, and a made-up rate."""
    answer = (
        "The Commercial Engines & Services (CES) operating profit grew by $8.6B in 4Q'25 compared to "
        "$7.7B in 4Q'24, representing a +10% year-over-year increase.\n\n[1]"
    )

    result = check_numbers(answer, {1: GE_CHART, 2: GE_FULL_YEAR})

    assert statuses(result) == {
        "$8.6B": ("other_source", [2]),
        "$7.7B": ("ok", [1]),
        "+10%": ("not_found", []),
    }
    assert result["summary"] == {"checked": 3, "ok": 1, "other_source": 1, "not_found": 1}


def test_correct_answer_passes():
    answer = "CES revenue grew to $9.5B in 4Q'25, up +24% y/y [1]."

    result = check_numbers(answer, {1: GE_CHART, 2: GE_FULL_YEAR})

    assert result["summary"] == {"checked": 2, "ok": 2, "other_source": 0, "not_found": 0}


@pytest.mark.parametrize(
    ("answer", "source"),
    [
        ("Revenue was 1350 [1].", "Revenue 1,350"),
        ("Margin 24% [1].", "margin 24.0%"),
        ("Revenue of 9.5 billion [1].", "$9.5B"),
        ("Cost fell by -420 bps [1].", "(420)bps"),
    ],
)
def test_the_same_value_written_differently_still_matches(answer, source):
    assert check_numbers(answer, {1: source})["summary"]["ok"] == 1


def test_a_percentage_must_be_a_percentage_in_the_source():
    result = check_numbers("Growth was 24% [1].", {1: "Revenue 24 units"})

    assert result["numbers"][0]["status"] == "not_found"


def test_sentence_without_citations_may_use_any_source():
    result = check_numbers("Revenue was $9.5B.", {1: "nothing here", 2: GE_CHART})

    assert result["numbers"][0] == {"text": "$9.5B", "status": "ok", "cited": [], "found_in": [2]}


def test_answer_without_numbers_has_nothing_to_check():
    assert check_numbers("I couldn't find this in your documents.", {1: "x"})["summary"]["checked"] == 0
