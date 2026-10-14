from lora.agent.notes import create_note
from lora.safety.policy import PolicyEngine


def test_creates_new_file_and_never_overwrites(tmp_path):
    a = create_note("My Plan", "hello", "md", folder=tmp_path)
    b = create_note("My Plan", "second", "md", folder=tmp_path)
    assert a["status"] == b["status"] == "success"
    assert a["path"] != b["path"]
    assert (tmp_path / "My Plan.md").read_text(encoding="utf-8").strip() == "hello"
    assert (tmp_path / "My Plan (2).md").exists()


def test_only_md_and_txt(tmp_path):
    for ext in ("exe", "bat", "py", "docx"):
        assert create_note("x", "y", ext, folder=tmp_path)["status"] == "error"
    assert not list(tmp_path.iterdir())


def test_filename_is_sanitised(tmp_path):
    r = create_note("../../evil:name*?", "x", "txt", folder=tmp_path)
    assert r["status"] == "success"
    assert (tmp_path / r["path"].replace("\\", "/").split("/")[-1]).parent == tmp_path


def test_empty_content_rejected(tmp_path):
    assert create_note("t", "   ", "md", folder=tmp_path)["status"] == "error"


def test_policy_is_enforced(tmp_path):
    policy = PolicyEngine(allowed_roots=[str(tmp_path / "allowed")])
    ok = create_note("n", "x", "md", folder=tmp_path / "allowed", policy=policy)
    bad = create_note("n", "x", "md", folder=tmp_path / "elsewhere", policy=policy)
    assert ok["status"] == "success"
    assert bad["status"] == "denied" and not (tmp_path / "elsewhere").exists()
