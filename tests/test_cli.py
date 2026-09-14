import json

from doccite.cli import main


def test_quality_gate_fails_on_false_answer(index_path, tmp_path, capsys):
    dataset = tmp_path / "synthetic.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "synthetic",
                "question": "What is the default timeout for network inactivity?",
                "answerable": False,
                "gold": [],
                "category": "deliberate mislabeled quality-gate test",
            }
        )
        + "\n"
    )
    output = tmp_path / "report.json"
    code = main(
        [
            "evaluate",
            "--index",
            str(index_path),
            "--dataset",
            str(dataset),
            "--output",
            str(output),
            "--max-false-answer-rate",
            "0.1",
        ]
    )
    assert code == 1
    assert json.loads(output.read_text())["metrics"]["false_answer_rate"] == 1
    assert output.with_suffix(".md").exists()


def test_missing_index_returns_actionable_error(tmp_path, capsys):
    assert main(["ask", "timeout", "--index", str(tmp_path / "missing.sqlite")]) == 2
    assert "doccite ingest" in capsys.readouterr().err
