from pathlib import Path

import pytest

from lora.storage.db import DatabaseManager


@pytest.mark.asyncio
async def test_database_init_and_wal_mode(temp_db_path: Path):
    db = DatabaseManager(temp_db_path)
    await db.initialize()

    # Verify WAL mode
    journal_mode = await db.get_journal_mode()
    assert journal_mode.lower() == "wal"

    await db.close()

@pytest.mark.asyncio
async def test_notebook_crud(temp_db_path: Path):
    db = DatabaseManager(temp_db_path)
    await db.initialize()

    # Create notebook
    nb_id = await db.create_notebook(
        title="Attention Mechanisms",
        origin="desktop",
        visibility="all"
    )
    assert nb_id is not None

    # Create desktop-only notebook from local files
    nb_local_id = await db.create_notebook(
        title="Local Thesis Notes",
        origin="desktop",
        visibility="desktop_only"
    )
    assert nb_local_id is not None

    # Retrieve notebooks filtered by visibility
    all_nbs = await db.list_notebooks(visibility="all")
    assert len(all_nbs) == 2

    mobile_nbs = await db.list_notebooks(visibility="mobile_visible")
    assert len(mobile_nbs) == 1
    assert mobile_nbs[0]["id"] == nb_id

    await db.close()

@pytest.mark.asyncio
async def test_sources_and_cards(temp_db_path: Path):
    db = DatabaseManager(temp_db_path)
    await db.initialize()

    nb_id = await db.create_notebook(title="Test", origin="desktop")

    src_id = await db.add_source(
        notebook_id=nb_id,
        url="https://arxiv.org/abs/2301.00001",
        title="Paper Title",
        tier=1,
        is_preprint=True,
        content_hash="hash123"
    )
    assert src_id is not None

    sources = await db.get_sources(notebook_id=nb_id)
    assert len(sources) == 1
    assert sources[0]["is_preprint"] is True

    # Add chunk and evidence card
    chunk_id = await db.add_chunk(
        source_id=src_id,
        page=1,
        section="Abstract",
        text="This is a test chunk containing key findings."
    )
    assert chunk_id is not None

    card_id = await db.add_evidence_card(
        chunk_id=chunk_id,
        quote="key findings",
        paraphrase="The study presented significant findings.",
        tier=1
    )
    assert card_id is not None

    cards = await db.get_evidence_cards(notebook_id=nb_id)
    assert len(cards) == 1
    assert cards[0]["quote"] == "key findings"

    await db.close()

@pytest.mark.asyncio
async def test_audit_log_record(temp_db_path: Path):
    db = DatabaseManager(temp_db_path)
    await db.initialize()

    log_id = await db.record_audit_event(
        task_id="task-001",
        action="open_path",
        tool="open_path",
        args={"path": "C:\\Thought\\Projects\\LORA\\README.md"},
        tier=0,
        gate_decision="allow",
        result="success"
    )
    assert log_id is not None

    events = await db.get_audit_events(task_id="task-001")
    assert len(events) == 1
    assert events[0]["gate_decision"] == "allow"

    await db.close()
