import json
import urllib.error
from unittest.mock import patch

import pytest

from doccite.answer import GenerationError, OpenAIGenerator, ask, validate_output
from doccite.ingest import DEFAULT_CORPUS


def test_answer_has_exact_quote_and_precise_permalink(index):
    result = ask(index, "What is the default timeout for network inactivity?")
    assert result.status == "answered"
    assert "five seconds" in result.claims[0].text
    for citation in result.citations:
        lines = (DEFAULT_CORPUS / citation.path).read_text().splitlines(keepends=True)
        assert citation.quote in "".join(lines[citation.start_line - 1 : citation.end_line])
        assert citation.url.endswith(f"#L{citation.start_line}-L{citation.end_line}")


def test_abstention_never_calls_generator(index):
    class MustNotRun:
        name = "must-not-run"

        def generate(self, question, hits):
            raise AssertionError("Weak evidence must not reach generation")

    result = ask(index, "What uptime SLA does HTTPX guarantee?", generator=MustNotRun())
    assert result.status == "abstained"
    assert not result.claims and not result.citations


@pytest.mark.parametrize(
    "payload",
    [
        {"answerable": True, "claims": []},
        {"answerable": "yes", "claims": []},
        {"answerable": False, "claims": [{"text": "sneaky claim"}]},
        {"answerable": True, "claims": [{"text": "no source", "evidence": []}]},
        {
            "answerable": True,
            "claims": [
                {
                    "text": "bad",
                    "evidence": [
                        {
                            "chunk_id": "invented",
                            "quote": "Invented source passage with enough length",
                        }
                    ],
                }
            ],
        },
    ],
)
def test_rejects_invalid_model_output(index, payload):
    with pytest.raises(GenerationError):
        validate_output(payload, index.search("timeout"))


def test_rejects_fabricated_quote_even_for_valid_source_id(index):
    hits = index.search("timeout")
    payload = {
        "answerable": True,
        "claims": [
            {
                "text": "Wrong default.",
                "evidence": [
                    {
                        "chunk_id": hits[0].chunk.id,
                        "quote": "The default timeout is exactly 999 seconds.",
                    }
                ],
            }
        ],
    }
    with pytest.raises(GenerationError, match="does not match"):
        validate_output(payload, hits)


def test_provider_abstention_is_preserved(index):
    class Abstainer:
        name = "test"

        def generate(self, question, hits):
            return {"answerable": False, "claims": []}

    result = ask(index, "What is the default timeout?", generator=Abstainer())
    assert result.status == "abstained" and result.reason == "model_abstention"


@pytest.mark.parametrize("question", ["", "  ", "a" * 2001])
def test_invalid_question(index, question):
    with pytest.raises(ValueError):
        ask(index, question)


def test_openai_request_and_response_contract(monkeypatch, index):
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-key")
    generator = OpenAIGenerator("test-model")
    from io import BytesIO

    output = {"answerable": False, "claims": []}
    body = {
        "status": "completed",
        "output": [
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": json.dumps(output)},
                ],
            }
        ],
    }
    with patch("urllib.request.urlopen", return_value=BytesIO(json.dumps(body).encode())) as call:
        assert generator.generate("timeout?", index.search("timeout")) == output
        request = call.call_args.args[0]
        assert request.full_url == "https://api.openai.com/v1/responses"
        payload = json.loads(request.data)
        assert payload["store"] is False
        assert payload["text"]["format"]["strict"] is True
        assert "untrusted data" in payload["instructions"]


def test_provider_outage_is_error_not_abstention(monkeypatch, index):
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-key")
    generator = OpenAIGenerator("test-model")
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("offline")):
        with pytest.raises(GenerationError):
            ask(index, "default timeout", generator=generator)
