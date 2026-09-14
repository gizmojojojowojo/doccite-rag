"""Compare search modes and sweep abstention thresholds using development data only."""

import argparse
import json
from pathlib import Path

from doccite.evaluate import evaluate
from doccite.search import SearchIndex


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path(".doccite/index.sqlite"))
    parser.add_argument("--output", type=Path, default=Path("reports/development-comparison.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    index = SearchIndex(args.index)
    reports = [
        evaluate(index, root / "benchmarks/dev.jsonl", mode=mode, threshold=threshold)
        for mode in ["bm25", "dense", "hybrid"]
        for threshold in [0.4, 0.5, 0.6, 0.62, 0.7, 0.8]
    ]
    output = {
        "dataset": "dev.jsonl",
        "note": "Development comparison only; no automatic tuning on test cases.",
        "runs": [{k: r[k] for k in ["config", "metrics", "dataset_sha256"]} for r in reports],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print("mode\tthreshold\tanswerable answer rate\tfalse answer rate\tgold evidence precision")
    for r in reports:
        c, m = r["config"], r["metrics"]
        print(
            f"{c['mode']}\t{c['threshold']}\t{m['answerable_answer_rate']}\t"
            f"{m['false_answer_rate']}\t{m['gold_evidence_answer_precision']}"
        )


if __name__ == "__main__":
    main()
