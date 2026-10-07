from lora.core.logging import (
    AuditLogger,
    scrub_sensitive_data,
)


def test_scrub_sensitive_data():
    raw_text = "Sending request with Authorization: Bearer secret_token_12345 and LORA_TOKEN=my_secret_token_999"
    scrubbed = scrub_sensitive_data(raw_text)
    assert "secret_token_12345" not in scrubbed
    assert "my_secret_token_999" not in scrubbed
    assert "[REDACTED]" in scrubbed

def test_scrub_dictionary():
    data = {
        "user": "alice",
        "token": "sensitive_token_abc",
        "api_key": "sk-1234567890abcdef",
        "path": "C:\\Thought\\Projects\\LORA\\test.pdf"
    }
    scrubbed = scrub_sensitive_data(data)
    assert scrubbed["user"] == "alice"
    assert scrubbed["token"] == "[REDACTED]"
    assert scrubbed["api_key"] == "[REDACTED]"
    assert scrubbed["path"] == "C:\\Thought\\Projects\\LORA\\test.pdf"

def test_audit_logger_format():
    logger = AuditLogger()
    event = logger.create_event(
        task_id="task-123",
        action="search",
        tool="web_search",
        args={"query": "RAG verification", "token": "secret"},
        tier=0,
        decision="allow"
    )
    assert event["task_id"] == "task-123"
    assert event["args"]["token"] == "[REDACTED]"
    assert event["decision"] == "allow"
