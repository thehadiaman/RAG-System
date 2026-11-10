import shutil

import pytest

from app.ingest.jobs import IngestQueue, document_dir
from app.store import documents
from app.store.db import Database
from tests.pdf_factory import build_text_pdf

pytestmark = pytest.mark.anyio


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "app.db")
    database.migrate()
    return database


def add_document(db, data_dir, status):
    doc_id = documents.new_id()
    folder = document_dir(data_dir, doc_id)
    folder.mkdir(parents=True)
    build_text_pdf(folder / "original.pdf")
    documents.create(db, doc_id, "a.pdf", doc_id, 100, 1)
    documents.set_status(db, doc_id, status)
    return doc_id


async def test_unfinished_documents_are_resumed_on_start(db, tmp_path):
    stuck = add_document(db, tmp_path, documents.EXTRACTING)
    done = add_document(db, tmp_path, documents.READY)
    queue = IngestQueue(db, tmp_path)

    queue.start()
    await queue.wait_until_idle()
    await queue.stop()

    assert documents.get(db, stuck)["status"] == documents.EXTRACTED
    assert len(documents.pages(db, stuck)) == 1
    assert documents.pages(db, done) == []  # finished documents are left alone


async def test_rereading_replaces_earlier_partial_results(db, tmp_path):
    doc_id = add_document(db, tmp_path, documents.QUEUED)
    queue = IngestQueue(db, tmp_path)

    await queue.process(doc_id)
    await queue.process(doc_id)

    assert len(documents.pages(db, doc_id)) == 1


async def test_document_deleted_midway_is_skipped_quietly(db, tmp_path):
    doc_id = add_document(db, tmp_path, documents.QUEUED)
    documents.delete(db, doc_id)
    shutil.rmtree(document_dir(tmp_path, doc_id))

    await IngestQueue(db, tmp_path).process(doc_id)  # must not raise

    assert documents.get(db, doc_id) is None
