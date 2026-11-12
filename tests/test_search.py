import pytest

from app.llm.embedder import EmbedderError
from app.search import Searcher, fts_query, fuse, keywords
from app.store.db import Database
from app.store.vectors import VectorStore

# ---- Keywords ----


def test_keywords_drop_filler_words_and_repeats():
    assert keywords("What was the revenue, and how much did Revenue grow?") == ["revenue", "grow"]


def test_keywords_keep_numbers_and_terms_whole():
    assert keywords("Was margin 12.5% in 4Q'25 or $4.2B?") == ["margin", "12.5%", "4Q'25", "$4.2B"]


def test_keywords_ignore_punctuation_and_query_syntax():
    # "and"/"or" are filler words; "NEAR" is kept, and quoting stops FTS5 reading it as an operator.
    assert keywords('revenue? (AND) "OR" * ^ NEAR ...') == ["revenue", "NEAR"]


def test_fts_query_quotes_every_word():
    assert fts_query(["revenue", "4Q'25", 'say "hi"']) == '"revenue" OR "4Q\'25" OR "say ""hi"""'


# ---- Fusion ----


def test_chunk_found_by_both_methods_comes_first():
    fused = fuse(meaning=["a", "b", "c"], keyword=["c", "d"])

    assert [chunk_id for chunk_id, *_ in fused] == ["c", "a", "b", "d"]
    c = fused[0]
    assert c[2:] == (3, 1)  # meaning rank 3, keyword rank 1
    assert c[1] == pytest.approx(1 / 63 + 1 / 61)


def test_chunk_found_by_one_method_is_kept():
    fused = dict((chunk_id, ranks) for chunk_id, _, *ranks in fuse(["a"], ["b"]))

    assert fused == {"a": [1, None], "b": [None, 1]}


def test_equal_scores_prefer_the_better_single_rank():
    assert [chunk_id for chunk_id, *_ in fuse(["a", "b"], ["b", "a"])] in (["a", "b"], ["b", "a"])
    assert fuse([], []) == []


# ---- Searcher with a real database and vector store ----

CHUNKS = {
    # id: (document, page, kind, section, text, vector)
    "a_rev": ("A", 1, "text", "Overview", "Revenue grew strongly across the Gulf region.", [1.0, 0.0, 0.0]),
    "a_ebitda": ("A", 2, "table", "Margins", "|Metric|2025|\n|---|---|\n|Adjusted EBITDA margin|18.4%|", [0.0, 1.0, 0.0]),
    "a_staff": ("A", 3, "text", "People", "Headcount fell slightly because of automation.", [0.0, 0.0, 1.0]),
    "b_sales": ("B", 1, "text", "Europe", "Sales rose in Europe.", [0.9, 0.1, 0.0]),
}


class StubEmbedder:
    """Returns a chosen vector for each question, or fails when `down` is set."""

    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = vectors
        self.down = False

    async def embed_query(self, question: str) -> list[float]:
        if self.down:
            raise EmbedderError("Embedding model nomic-embed-text isn't installed. Run: ollama pull nomic-embed-text")
        return self.vectors.get(question, [0.3, 0.3, 0.3])


@pytest.fixture
def store(tmp_path):
    db = Database(tmp_path / "app.db")
    db.migrate()
    vectors = VectorStore(tmp_path / "chroma")
    with db.session() as conn:
        for doc_id, name in (("A", "annual-report.pdf"), ("B", "europe.pdf")):
            conn.execute(
                "INSERT INTO documents (id, filename, sha256, size_bytes, status) VALUES (?, ?, ?, 1, 'ready')",
                (doc_id, name, doc_id),
            )
        for position, (chunk_id, (doc_id, page_no, kind, section, text, _)) in enumerate(CHUNKS.items()):
            conn.execute(
                """INSERT INTO chunks (id, document_id, position, page_no, kind, section, text)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (chunk_id, doc_id, position, page_no, kind, section, text),
            )
    vectors.upsert(
        list(CHUNKS),
        [c[5] for c in CHUNKS.values()],
        [{"document_id": c[0], "page_no": c[1], "kind": c[2]} for c in CHUNKS.values()],
    )
    return db, vectors


def searcher(store, question_vectors=None, candidates=30):
    db, vectors = store
    return Searcher(db, vectors, StubEmbedder(question_vectors or {}), candidates=candidates)


def ids(result):
    return [hit.chunk["id"] for hit in result.hits]


@pytest.mark.anyio
async def test_chunk_found_by_meaning_and_keywords_ranks_first(store):
    result = await searcher(store, {"How did sales do?": [1.0, 0.0, 0.0]}).search("How did sales do?")

    assert result.mode == "hybrid"
    assert ids(result)[:2] == ["b_sales", "a_rev"]
    top = result.hits[0]
    assert top.found_by == ["meaning", "keyword"]
    assert top.chunk["filename"] == "europe.pdf"
    assert top.distance == pytest.approx(1 - 0.9 / (0.9**2 + 0.1**2) ** 0.5, abs=1e-4)


@pytest.mark.anyio
async def test_exact_number_is_found_by_keywords(store):
    # Meaning search points at the wrong chunk; the exact figure still wins.
    result = await searcher(store, {"18.4%": [0.0, 0.0, 1.0]}).search("18.4%")

    assert ids(result)[0] == "a_ebitda"
    assert result.hits[0].keyword_rank == 1
    assert result.keywords == ["18.4%"]


@pytest.mark.anyio
async def test_differently_worded_answer_is_found_by_meaning(store):
    result = await searcher(store, {"workforce reduction": [0.0, 0.0, 1.0]}, candidates=1).search("workforce reduction")

    assert ids(result) == ["a_staff"]
    assert result.hits[0].found_by == ["meaning"]


@pytest.mark.anyio
async def test_search_can_be_limited_to_some_documents(store):
    result = await searcher(store).search("revenue sales growth", document_ids=["B"])

    assert ids(result) == ["b_sales"]


@pytest.mark.anyio
async def test_limit_caps_the_number_of_hits(store):
    assert len((await searcher(store).search("revenue", limit=2)).hits) == 2


@pytest.mark.anyio
async def test_falls_back_to_keywords_when_the_embedder_is_down(store):
    search = searcher(store)
    search.embedder.down = True

    result = await search.search("EBITDA margin")

    assert result.mode == "keyword_only"
    assert "ollama pull nomic-embed-text" in result.notice
    assert ids(result) == ["a_ebitda"]
    assert result.hits[0].found_by == ["keyword"]


@pytest.mark.anyio
async def test_odd_characters_in_the_question_are_harmless(store):
    result = await searcher(store).search('revenue? "AND" OR (NOT) * ^ NEAR {x} [y] -z +w')

    assert "a_rev" in ids(result)


@pytest.mark.anyio
async def test_question_of_only_filler_words_still_uses_meaning_search(store):
    result = await searcher(store, {"what is the": [0.0, 1.0, 0.0]}).search("what is the")

    assert result.keywords == []
    assert ids(result)[0] == "a_ebitda"


@pytest.mark.anyio
async def test_vectors_without_a_chunk_are_skipped(store):
    db, vectors = store
    vectors.upsert(["gone"], [[1.0, 0.0, 0.0]], [{"document_id": "A", "page_no": 9, "kind": "text"}])

    result = await searcher(store, {"q": [1.0, 0.0, 0.0]}).search("q")

    assert "gone" not in ids(result)
    assert ids(result)[0] == "a_rev"
