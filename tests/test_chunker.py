from app.ingest.chunker import (
    MAX_TABLE_CHARS,
    MAX_TEXT_CHARS,
    Chunk,
    chunk_document,
    chunk_page,
    embedding_text,
    figure_chunk,
    split_blocks,
    split_table,
)


def page(page_no, text):
    return {"page_no": page_no, "text": text}


def figure(page_no=1, **fields):
    return {
        "id": f"fig{page_no}",
        "page_no": page_no,
        "image_path": f"documents/d/figures/p{page_no}.png",
        "kind": None,
        "description": None,
        "caption": "",
        "text_inside": "",
        **fields,
    }


SENTENCE = "Revenue grew strongly in the Gulf region this year. "


# ---- Blocks ----


def test_markdown_is_split_into_headings_paragraphs_and_tables():
    markdown = "# Results\n\nFirst paragraph\nsecond line\n\n|A|B|\n|---|---|\n|1|2|\n\n[Picture text: 47%]\n\nLast."

    assert [(b.kind, b.text) for b in split_blocks(markdown)] == [
        ("heading", "# Results"),
        ("text", "First paragraph\nsecond line"),
        ("table", "|A|B|\n|---|---|\n|1|2|"),
        ("picture_text", "[Picture text: 47%]"),
        ("text", "Last."),
    ]


# ---- Text ----


def test_short_page_becomes_one_chunk_under_its_heading():
    chunks, section = chunk_page(1, "# Overview\n\nWe had a good year.\n\nMargins improved.", "")

    assert len(chunks) == 1
    assert chunks[0].kind == "text"
    assert chunks[0].section == "Overview"
    assert chunks[0].text == "# Overview\n\nWe had a good year.\n\nMargins improved."
    assert section == "Overview"


def test_each_heading_starts_a_new_chunk():
    chunks, _ = chunk_page(1, "# Revenue\n\nUp 12%.\n\n## Costs\n\nDown 3%.", "")

    assert [(c.section, c.text) for c in chunks] == [
        ("Revenue", "# Revenue\n\nUp 12%."),
        ("Costs", "## Costs\n\nDown 3%."),
    ]


def test_long_text_is_packed_into_chunks_that_overlap():
    paragraphs = "\n\n".join(f"Paragraph {i}. " + SENTENCE * 8 for i in range(8))

    chunks, _ = chunk_page(1, "# Report\n\n" + paragraphs, "")

    assert len(chunks) > 1
    assert all(len(c.text) <= MAX_TEXT_CHARS + 250 for c in chunks)
    assert all(c.section == "Report" for c in chunks)
    # The end of one chunk is repeated at the start of the next.
    assert chunks[1].text.startswith(SENTENCE.strip())
    for i in range(8):
        assert any(f"Paragraph {i}." in c.text for c in chunks)


def test_one_huge_paragraph_is_cut_at_sentence_ends():
    chunks, _ = chunk_page(1, SENTENCE * 100, "")

    assert len(chunks) > 2
    assert all(c.text.endswith(".") for c in chunks)


def test_section_names_are_plain_text():
    chunks, _ = chunk_page(1, "## **Backlog**<sup>**1**</sup>\n\nUp 5.6%.\n\n# FCF* conversion<sup>-a)</sup>\n\nOver 100%.", "")

    assert [c.section for c in chunks] == ["Backlog", "FCF* conversion"]


def test_section_carries_over_from_the_previous_page():
    chunks, section = chunk_page(2, "Continued text without a heading.", "Risk factors")

    assert chunks[0].section == "Risk factors"
    assert section == "Risk factors"


def test_heading_with_nothing_under_it_is_not_a_chunk_but_names_what_follows():
    chunks, _ = chunk_page(1, "# Results by segment\n\n|A|B|\n|---|---|\n|1|2|", "")

    assert [(c.kind, c.section) for c in chunks] == [("table", "Results by segment")]


def test_page_with_only_a_title_is_kept():
    chunks, _ = chunk_page(1, "# Investor Presentation", "")

    assert [c.text for c in chunks] == ["# Investor Presentation"]


def test_picture_text_is_kept_unless_asked_to_drop_it():
    markdown = "# Mix\n\n[Picture text: Retail 47% · Online 19%]"

    kept, _ = chunk_page(1, markdown, "")
    dropped, _ = chunk_page(1, markdown, "", drop_picture_text=True, keep_lone_headings=False)

    assert "Retail 47%" in kept[0].text
    assert dropped == []


# ---- Tables ----


def test_small_table_stays_whole_and_separate_from_text():
    chunks, _ = chunk_page(1, "Intro.\n\n|Segment|2025|\n|---|---|\n|Retail|1,350|\n\nOutro.", "Results")

    assert [c.kind for c in chunks] == ["text", "table", "text"]
    assert chunks[1].text == "|Segment|2025|\n|---|---|\n|Retail|1,350|"


def test_big_table_is_split_by_rows_with_the_header_repeated():
    header = "|Item|Value|\n|---|---|"
    rows = [f"|Line item number {i}|{i * 1000:,}|" for i in range(200)]
    table = header + "\n" + "\n".join(rows)

    pieces = split_table(table)

    assert len(pieces) > 1
    assert all(piece.startswith(header) for piece in pieces)
    assert all(len(piece) <= MAX_TABLE_CHARS for piece in pieces)
    assert sum(piece.count("|Line item number") for piece in pieces) == 200


# ---- Figures ----


def test_described_figure_keeps_description_caption_and_pdf_text():
    chunk = figure_chunk(
        figure(3, kind="chart", description="Type: chart\nTitle: Revenue", caption="Figure 1: Revenue", text_inside="2024 $35.1B"),
        "Guidance",
    )

    assert chunk.kind == "figure"
    assert chunk.section == "Guidance"
    assert chunk.figure_id == "fig3"
    assert chunk.image_path == "documents/d/figures/p3.png"
    assert chunk.text == (
        "Type: chart\nTitle: Revenue\n\nCaption: Figure 1: Revenue\n\nText in the figure (from the PDF): 2024 $35.1B"
    )


def test_decorative_figures_are_left_out():
    assert figure_chunk(figure(kind="decorative", description="Type: decorative\nTitle: Logo"), "") is None


def test_figure_never_read_by_a_vision_model_still_uses_its_pdf_text():
    chunk = figure_chunk(figure(caption="Figure 2: Margin by year", text_inside="2023 24% 2024 28%"), "")

    assert chunk.text.startswith("Figure (not read by a vision model).")
    assert "2024 28%" in chunk.text


def test_figure_with_nothing_to_say_is_left_out():
    assert figure_chunk(figure(), "") is None


# ---- Whole document ----


def test_document_chunks_come_in_page_order_with_figures_after_page_text():
    pages = [page(1, "# Intro\n\nHello."), page(2, "# Revenue trend\n\nSee the chart.")]
    figures = [figure(2, kind="chart", description="Type: chart\nTitle: Trend")]

    chunks = chunk_document(pages, figures)

    assert [(c.page_no, c.kind, c.section) for c in chunks] == [
        (1, "text", "Intro"),
        (2, "text", "Revenue trend"),
        (2, "figure", "Revenue trend"),
    ]


def test_figure_section_is_plain_text_too():
    chunks = chunk_document(
        [page(1, "# **Backlog**<sup>1</sup>\n\nSee chart.")], [figure(1, kind="chart", description="Type: chart")]
    )

    assert [c.section for c in chunks] == ["Backlog", "Backlog"]


def test_picture_text_is_dropped_only_when_the_page_figures_are_described():
    text = "# Mix\n\n[Picture text: Retail 47%]"
    described = chunk_document([page(1, text)], [figure(1, kind="chart", description="Type: chart")])
    undescribed = chunk_document([page(1, text)], [figure(1, text_inside="Retail 47% Online 19% Other 34%")])

    assert [c.kind for c in described] == ["figure"]
    assert [c.kind for c in undescribed] == ["text", "figure"]


def test_embedding_text_leaves_out_the_unread_figure_placeholder():
    chunk = figure_chunk(figure(caption="Figure 2: Margin by year", text_inside="2023 24% 2024 28%"), "Margins")

    assert embedding_text("report.pdf", chunk) == (
        "report › Margins\nCaption: Figure 2: Margin by year\n\nText in the figure (from the PDF): 2023 24% 2024 28%"
    )


def test_embedding_text_says_where_the_chunk_is_from():
    chunk = Chunk(1, "text", "Revenue", "Up 12%.")

    assert embedding_text("annual-report.pdf", chunk) == "annual-report › Revenue\nUp 12%."
    assert embedding_text("annual-report.pdf", Chunk(1, "text", "", "Up 12%.")) == "annual-report\nUp 12%."
