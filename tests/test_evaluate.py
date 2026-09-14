import json

import pytest

from doccite.answer import GenerationError
from doccite.evaluate import evaluate, ratio


def write_cases(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_errors_are_not_counted_as_abstentions(index, tmp_path):
    class BrokenProvider:
        name = "broken"

        def generate(self, question, hits):
            raise GenerationError("provider offline")

    dataset = write_cases(
        tmp_path / "data.jsonl",
        [
            {
                "id": "unanswerable",
                "question": "default timeout",
                "answerable": False,
                "gold": [],
                "category": "synthetic error-accounting test",
            }
        ],
    )
    metrics = evaluate(index, dataset, generator=BrokenProvider())["metrics"]
    assert metrics["errors"] == 1
    assert metrics["abstention_recall"] == 0
    assert metrics["abstention_precision"] is None


def test_gold_quote_must_exist_in_corpus(index, tmp_path):
    dataset = write_cases(
        tmp_path / "bad.jsonl",
        [
            {
                "id": "invalid",
                "question": "timeout",
                "answerable": True,
                "gold": [
                    {"path": "docs/quickstart.md", "quote": "fabricated source that does not exist"}
                ],
                "category": "invalid",
            }
        ],
    )
    with pytest.raises(ValueError, match="Gold passage missing"):
        evaluate(index, dataset)


def test_undefined_metrics_do_not_look_like_perfect_scores():
    assert ratio(0, 0) is None
    assert ratio(3, 4) == 0.75


def test_provider_error_does_not_erase_retrieval_success(index, tmp_path):
    class BrokenProvider:
        name = "broken"

        def generate(self, question, hits):
            raise GenerationError("provider offline")

    dataset = write_cases(
        tmp_path / "positive.jsonl",
        [
            {
                "id": "positive",
                "question": "What is the default timeout for network inactivity?",
                "answerable": True,
                "category": "synthetic error-accounting test",
                "gold": [
                    {
                        "path": "docs/quickstart.md",
                        "quote": "The default timeout for network inactivity is five seconds.",
                    }
                ],
            }
        ],
    )
    metrics = evaluate(index, dataset, generator=BrokenProvider())["metrics"]
    assert metrics["errors"] == 1
    assert metrics["retrieval_hit_rate_at_k"] == 1
    assert metrics["answerable_answer_rate"] == 0


def test_valid_citation_is_not_automatically_gold_evidence(index, tmp_path):
    # This is a real passage, deliberately assigned to the wrong question.
    dataset = write_cases(
        tmp_path / "wrong-gold.jsonl",
        [
            {
                "id": "wrong-gold",
                "question": "What is the default timeout for network inactivity?",
                "answerable": True,
                "category": "synthetic metric test",
                "gold": [
                    {
                        "path": "docs/advanced/clients.md",
                        "quote": "the `Client` will reuse the underlying TCP connection",
                    }
                ],
            }
        ],
    )
    metrics = evaluate(index, dataset)["metrics"]
    assert metrics["citation_validity"] == 1
    assert metrics["gold_evidence_answer_precision"] == 0
