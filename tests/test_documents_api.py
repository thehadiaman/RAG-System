import time

import pytest

from app.config import get_settings
from tests.fake_bedrock import FakeBedrock
from tests.pdf_factory import build_password_pdf, build_sample_pdf, build_text_pdf


def upload(client, path, name=None, extractor=None):
    data = {"extractor": extractor} if extractor else {}
    with path.open("rb") as f:
        return client.post("/api/documents", files={"file": (name or path.name, f, "application/pdf")}, data=data)


def wait_for(client, doc_id, statuses=("ready", "extracted", "failed"), timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        doc = client.get(f"/api/documents/{doc_id}").json()
        if doc["status"] in statuses:
            return doc
        time.sleep(0.1)
    raise AssertionError(f"document stuck in {doc['status']}")


@pytest.fixture
def sample_pdf(tmp_path):
    return build_sample_pdf(tmp_path / "annual-report.pdf")


def test_upload_is_read_in_the_background(client, sample_pdf):
    response = upload(client, sample_pdf)

    assert response.status_code == 201
    created = response.json()
    assert created["filename"] == "annual-report.pdf"
    assert created["page_count"] == 5
    assert created["duplicate"] is False

    doc = wait_for(client, created["id"])
    assert doc["status"] == "ready"
    assert doc["progress"] == {"done": 5, "total": 5, "note": None}

    pages = {p["page_no"]: p for p in doc["pages"]}
    assert pages[2]["table_count"] == 1
    assert pages[3]["figure_count"] == 1
    assert pages[4]["figure_count"] == 1
    assert pages[5]["needs_ocr"] is True
    assert [f["page_no"] for f in doc["figures"]] == [3, 4]


def test_page_text_and_images_can_be_fetched(client, sample_pdf):
    doc_id = upload(client, sample_pdf).json()["id"]
    doc = wait_for(client, doc_id)

    page = client.get(f"/api/documents/{doc_id}/pages/2").json()
    assert "|Retail|1,200|1,350|+12.5%|" in page["text"]

    preview = client.get(page["image_url"])
    assert preview.status_code == 200
    assert preview.headers["content-type"] == "image/jpeg"

    figure = client.get(doc["figures"][0]["image_url"])
    assert figure.status_code == 200
    assert figure.headers["content-type"] == "image/png"


def test_same_file_twice_returns_the_first_upload(client, sample_pdf):
    first = upload(client, sample_pdf).json()

    second = upload(client, sample_pdf, name="copy.pdf")

    assert second.status_code == 200
    assert second.json()["id"] == first["id"]
    assert second.json()["duplicate"] is True
    assert len(client.get("/api/documents").json()) == 1


def test_rejects_files_that_are_not_pdfs(client, tmp_path):
    path = tmp_path / "notes.pdf"
    path.write_text("not a pdf")

    response = upload(client, path)

    assert response.status_code == 400
    assert response.json()["detail"] == "This file isn't a PDF."


def test_rejects_password_protected_pdfs(client, tmp_path):
    response = upload(client, build_password_pdf(tmp_path / "locked.pdf"))

    assert response.status_code == 400
    assert "password protected" in response.json()["detail"]


def test_rejects_empty_files(client, tmp_path):
    path = tmp_path / "empty.pdf"
    path.write_bytes(b"")

    assert upload(client, path).status_code == 400


def test_rejects_files_over_the_size_limit(client, tmp_path, monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "0")
    get_settings.cache_clear()

    response = upload(client, build_text_pdf(tmp_path / "small.pdf"))

    assert response.status_code == 413
    assert list((get_settings().data_dir / "documents").iterdir()) == []  # temp file cleaned up


def test_delete_removes_the_document_and_its_files(client, sample_pdf, data_dir):
    doc_id = upload(client, sample_pdf).json()["id"]
    wait_for(client, doc_id)
    assert (data_dir / "documents" / doc_id).is_dir()

    assert client.delete(f"/api/documents/{doc_id}").status_code == 204

    assert client.get(f"/api/documents/{doc_id}").status_code == 404
    assert not (data_dir / "documents" / doc_id).exists()
    assert client.delete(f"/api/documents/{doc_id}").status_code == 404


def test_a_failed_read_is_reported_on_the_document(client, sample_pdf, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("the PDF parser crashed")

    monkeypatch.setattr("app.ingest.jobs.extract_document", broken)

    doc = wait_for(client, upload(client, sample_pdf).json()["id"])

    assert doc["status"] == "failed"
    assert doc["error"] == "the PDF parser crashed"


def test_malformed_ids_are_rejected(client):
    assert client.get("/api/documents/../../etc").status_code == 404
    assert client.get("/api/documents/not-an-id").status_code == 422


# ---- Vision step ----


def test_extractors_lists_what_can_be_used(client):
    data = client.get("/api/extractors").json()

    assert data["default"] == "local"
    by_id = {e["id"]: e for e in data["extractors"]}
    assert by_id["local"]["available"] is True
    assert by_id["none"]["available"] is True
    assert by_id["aws"]["available"] is False
    assert "AWS keys are not set" in by_id["aws"]["problem"]


def test_upload_reads_figures_and_scanned_pages_with_the_local_model(client, sample_pdf):
    doc = wait_for(client, upload(client, sample_pdf).json()["id"])

    assert doc["extractor"] == "local"
    assert doc["warning"] is None
    assert [f["kind"] for f in doc["figures"]] == ["chart", "chart"]
    assert all(f["described_by"] == "qwen2.5vl:3b" for f in doc["figures"])
    page = client.get(f"/api/documents/{doc['id']}/pages/5").json()
    assert page["text_source"] == "vision"


def test_text_only_upload_skips_the_vision_step(client, sample_pdf, fake_ollama):
    doc = wait_for(client, upload(client, sample_pdf, extractor="none").json()["id"])

    assert doc["extractor"] == "none"
    assert all(f["description"] is None for f in doc["figures"])
    assert fake_ollama.calls_to("/api/chat") == []


def test_upload_refuses_an_extractor_that_is_not_set_up(client, sample_pdf):
    response = upload(client, sample_pdf, extractor="aws")

    assert response.status_code == 400
    assert "AWS keys are not set" in response.json()["detail"]
    assert client.get("/api/documents").json() == []


def test_upload_refuses_an_unknown_extractor(client, sample_pdf):
    assert upload(client, sample_pdf, extractor="gpt").status_code == 422


@pytest.mark.parametrize("fake_bedrock", [FakeBedrock(configured=True)])
def test_vision_step_can_be_run_again_with_another_model(client, sample_pdf):
    doc_id = upload(client, sample_pdf, extractor="local").json()["id"]
    wait_for(client, doc_id)

    response = client.post(f"/api/documents/{doc_id}/describe", params={"extractor": "aws"})

    assert response.status_code == 202
    assert response.json()["status"] == "describing"
    doc = wait_for(client, doc_id)
    assert doc["extractor"] == "aws"
    assert {f["described_by"] for f in doc["figures"]} == {"amazon.nova-lite-v1:0"}


# ---- Search index ----


def test_ready_document_reports_its_chunks(client, sample_pdf):
    doc = wait_for(client, upload(client, sample_pdf, extractor="none").json()["id"])

    assert doc["status"] == "ready"
    assert doc["chunk_counts"]["table"] == 1
    assert doc["chunk_counts"]["figure"] == 2
    assert doc["chunk_counts"]["text"] >= 3
    health = client.get("/api/health").json()
    assert health["components"]["vectors"]["detail"] == f"{sum(doc['chunk_counts'].values())} chunks"


def test_chunks_can_be_listed_and_filtered(client, sample_pdf):
    doc_id = upload(client, sample_pdf, extractor="none").json()["id"]
    wait_for(client, doc_id)

    tables = client.get(f"/api/documents/{doc_id}/chunks", params={"kind": "table"}).json()
    assert len(tables) == 1
    assert tables[0]["page_no"] == 2
    assert tables[0]["section"] == "Results by segment"
    assert "|Retail|1,200|1,350|+12.5%|" in tables[0]["text"]

    page3 = client.get(f"/api/documents/{doc_id}/chunks", params={"page_no": 3}).json()
    figure = [c for c in page3 if c["kind"] == "figure"][0]
    assert "Caption: Figure 1: Revenue by year (AED million)" in figure["text"]
    assert client.get(figure["image_url"]).status_code == 200


def test_index_can_be_rebuilt_without_duplicates(client, sample_pdf):
    doc_id = upload(client, sample_pdf, extractor="none").json()["id"]
    before = wait_for(client, doc_id)["chunk_counts"]

    response = client.post(f"/api/documents/{doc_id}/index")

    assert response.status_code == 202
    assert wait_for(client, doc_id, statuses=("ready",))["chunk_counts"] == before
    total = sum(before.values())
    assert client.get("/api/health").json()["components"]["vectors"]["detail"] == f"{total} chunks"


def test_delete_also_empties_the_search_index(client, sample_pdf):
    doc_id = upload(client, sample_pdf, extractor="none").json()["id"]
    wait_for(client, doc_id)

    client.delete(f"/api/documents/{doc_id}")

    assert client.get("/api/health").json()["components"]["vectors"]["detail"] == "0 chunks"


# ---- Search ----


def test_search_finds_the_table_with_links_to_the_page(client, sample_pdf):
    doc_id = upload(client, sample_pdf, extractor="none").json()["id"]
    wait_for(client, doc_id)

    response = client.post("/api/search", json={"question": "Wholesale results by segment"})

    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "hybrid"
    assert data["keywords"] == ["Wholesale", "results", "segment"]
    top = data["hits"][0]
    assert top["kind"] == "table"
    assert top["page_no"] == 2
    assert top["filename"] == "annual-report.pdf"
    assert "keyword" in top["found_by"]
    assert client.get(top["page_image_url"]).status_code == 200


def test_search_returns_figure_image_links(client, sample_pdf):
    doc_id = upload(client, sample_pdf, extractor="none").json()["id"]
    wait_for(client, doc_id)

    hits = client.post("/api/search", json={"question": "Revenue by year AED million", "limit": 30}).json()["hits"]

    figure = [h for h in hits if h["kind"] == "figure" and h["page_no"] == 3][0]
    assert client.get(figure["figure_image_url"]).status_code == 200


def test_search_checks_its_input(client, sample_pdf):
    assert client.post("/api/search", json={"question": "   "}).status_code == 422
    assert client.post("/api/search", json={"question": "x", "limit": 0}).status_code == 422
    assert client.post("/api/search", json={"question": "x", "document_ids": ["../x"]}).status_code == 422

    response = client.post("/api/search", json={"question": "x", "document_ids": ["0" * 32]})
    assert response.status_code == 404
    assert "Unknown document" in response.json()["detail"]


def test_search_with_nothing_indexed_returns_no_hits(client):
    data = client.post("/api/search", json={"question": "revenue"}).json()

    assert data["hits"] == []
