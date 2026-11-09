import time

import pytest

from app.config import get_settings
from tests.pdf_factory import build_password_pdf, build_sample_pdf, build_text_pdf


def upload(client, path, name=None):
    with path.open("rb") as f:
        return client.post("/api/documents", files={"file": (name or path.name, f, "application/pdf")})


def wait_for(client, doc_id, statuses=("extracted", "failed"), timeout=60):
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
    assert doc["status"] == "extracted"
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
