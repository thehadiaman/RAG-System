import pytest

from app.store.vectors import VectorStore, collection_for


@pytest.fixture
def store(tmp_path):
    return VectorStore(tmp_path / "chroma")


def add_samples(store):
    store.upsert(
        ids=["a1", "a2", "b1"],
        embeddings=[[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 0.0, 1.0]],
        metadatas=[
            {"document_id": "A", "page_no": 1, "kind": "text"},
            {"document_id": "A", "page_no": 2, "kind": "table"},
            {"document_id": "B", "page_no": 1, "kind": "figure"},
        ],
    )


def test_empty_store_finds_nothing(store):
    assert store.count() == 0
    assert store.query([1.0, 0.0, 0.0]) == []


def test_closest_chunk_comes_first(store):
    add_samples(store)

    hits = store.query([1.0, 0.0, 0.0], limit=3)

    assert [h.chunk_id for h in hits] == ["a1", "a2", "b1"]
    assert hits[0].distance == pytest.approx(0.0, abs=1e-5)
    assert hits[0].metadata == {"document_id": "A", "page_no": 1, "kind": "text"}


def test_search_can_be_limited_to_some_documents(store):
    add_samples(store)

    hits = store.query([1.0, 0.0, 0.0], document_ids=["B"])

    assert [h.chunk_id for h in hits] == ["b1"]


def test_upsert_replaces_a_chunk_with_the_same_id(store):
    add_samples(store)

    store.upsert(ids=["a1"], embeddings=[[0.0, 1.0, 0.0]], metadatas=[{"document_id": "A", "page_no": 1}])

    assert store.count() == 3
    assert store.query([0.0, 1.0, 0.0], limit=1)[0].chunk_id == "a1"


def test_deleting_a_document_removes_only_its_chunks(store):
    add_samples(store)

    store.delete_document("A")

    assert store.count() == 1
    assert [h.chunk_id for h in store.query([1.0, 0.0, 0.0])] == ["b1"]


def test_vectors_from_an_earlier_search_model_are_removed(tmp_path):
    old = VectorStore(tmp_path / "chroma")  # before Phase 14 the collection had no model in its name
    add_samples(old)
    add_samples(VectorStore(tmp_path / "chroma", collection_for("model-a")))

    current = VectorStore(tmp_path / "chroma", collection_for("model-b"))
    add_samples(current)

    assert sorted(current.remove_other_collections()) == ["chunks", "chunks-model-a"]
    assert current.count() == 3
    assert VectorStore(tmp_path / "chroma", collection_for("model-a")).count() == 0
