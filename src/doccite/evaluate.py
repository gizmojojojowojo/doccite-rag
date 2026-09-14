"""Reproducible passage-level retrieval and selective-answering evaluation."""

import hashlib
import json
import platform
import statistics
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from doccite import __version__
from doccite.answer import GenerationError, Generator, ask
from doccite.search import SearchIndex


def runtime_fingerprint() -> dict:
    versions = {}
    for name in ["sentence-transformers", "transformers", "torch"]:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            pass
    source = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        source.update(path.name.encode())
        source.update(path.read_bytes())
    return {"source_sha256": source.hexdigest(), "installed_optional_packages": versions}


def ratio(n: float, d: int) -> float | None:
    return round(n / d, 4) if d else None


def evaluate(
    index: SearchIndex,
    dataset: Path,
    *,
    generator: Generator | None = None,
    threshold: float = 0.62,
    mode: str = "hybrid",
    k: int = 5,
) -> dict:
    data = dataset.read_bytes()
    cases = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Evaluation dataset must contain unique, nonempty cases")
    rows = []
    for case in cases:
        if type(case.get("answerable")) is not bool or not isinstance(case.get("gold"), list):
            raise ValueError(f"Invalid benchmark schema: {case['id']}")
        if case["answerable"] and not case["gold"]:
            raise ValueError(f"Answerable case needs gold evidence: {case['id']}")
        if not case["answerable"] and case["gold"]:
            raise ValueError(f"Unanswerable case must not have gold evidence: {case['id']}")
        gold_ids: set[str] = set()
        for gold in case["gold"]:
            matching = [
                c for c in index.chunks if c.path == gold["path"] and gold["quote"] in c.text
            ]
            if not matching or len(gold["quote"]) < 20:
                raise ValueError(f"Gold passage missing from indexed corpus: {case['id']}")
            gold_ids.update(c.id for c in matching)
        try:
            result = ask(
                index, case["question"], generator=generator, threshold=threshold, mode=mode, k=k
            )
        except GenerationError as exc:
            # Recheck deterministic retrieval after a provider failure so a generation
            # outage does not get mislabeled as a retrieval miss.
            error_hits = index.search(case["question"], k=k, mode=mode)
            error_ranks = [n for n, hit in enumerate(error_hits, 1) if hit.chunk.id in gold_ids]
            rows.append(
                {
                    "id": case["id"],
                    "question": case["question"],
                    "expected_answerable": case["answerable"],
                    "category": case["category"],
                    "status": "error",
                    "error": str(exc),
                    "retrieval_hit": bool(error_ranks) if case["answerable"] else None,
                    "reciprocal_rank": 1 / min(error_ranks) if error_ranks else 0.0,
                }
            )
            continue
        ranks = [n for n, h in enumerate(result.trace["retrieved"], 1) if h["chunk_id"] in gold_ids]
        # Citation relevance uses the actual quoted span, not just a gold document/chunk ID.
        relevant = sum(
            any(c.path == g["path"] and g["quote"] in c.quote for g in case["gold"])
            for c in result.citations
        )
        rows.append(
            {
                "id": case["id"],
                "question": case["question"],
                "category": case["category"],
                "expected_answerable": case["answerable"],
                "status": result.status,
                "reason": result.reason,
                "retrieval_hit": bool(ranks) if case["answerable"] else None,
                "reciprocal_rank": 1 / min(ranks) if ranks else 0.0,
                "citation_count": len(result.citations),
                "gold_supported_citations": relevant,
                "citation_validity": all(
                    c.quote in next(ch.text for ch in index.chunks if ch.id == c.chunk_id)
                    for c in result.citations
                )
                if result.citations
                else None,
                "answer": result.to_dict(),
            }
        )
    positives = [r for r in rows if r["expected_answerable"]]
    negatives = [r for r in rows if not r["expected_answerable"]]
    answered = [r for r in rows if r["status"] == "answered"]
    abstained = [r for r in rows if r["status"] == "abstained"]
    false_answers = sum(r["status"] == "answered" for r in negatives)
    true_abstentions = sum(r["status"] == "abstained" for r in negatives)
    supported = sum(
        r["expected_answerable"] and r.get("gold_supported_citations", 0) > 0 for r in answered
    )
    citation_count = sum(r.get("citation_count", 0) for r in rows)
    latencies = [r["answer"]["trace"]["total_ms"] for r in rows if "answer" in r]
    metrics = {
        "cases": len(rows),
        "answerable_cases": len(positives),
        "unanswerable_cases": len(negatives),
        "errors": sum(r["status"] == "error" for r in rows),
        "retrieval_hit_rate_at_k": ratio(
            sum(bool(r.get("retrieval_hit")) for r in positives), len(positives)
        ),
        "mrr_at_k": ratio(sum(r.get("reciprocal_rank", 0) for r in positives), len(positives)),
        "answerable_answer_rate": ratio(
            sum(r["status"] == "answered" for r in positives), len(positives)
        ),
        "abstention_recall": ratio(true_abstentions, len(negatives)),
        "abstention_precision": ratio(true_abstentions, len(abstained)),
        "false_answer_rate": ratio(false_answers, len(negatives)),
        "answer_coverage": ratio(len(answered), len(rows)),
        "gold_evidence_answer_precision": ratio(supported, len(answered)),
        "citation_validity": ratio(
            sum(r.get("citation_count", 0) for r in rows if r.get("citation_validity")),
            citation_count,
        ),
        "gold_quote_citation_precision": ratio(
            sum(r.get("gold_supported_citations", 0) for r in rows), citation_count
        ),
        "latency_p50_ms": round(statistics.median(latencies), 3) if latencies else None,
    }
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "project_version": __version__,
        "python": platform.python_version(),
        "runtime": runtime_fingerprint(),
        "dataset": dataset.name,
        "dataset_sha256": hashlib.sha256(data).hexdigest(),
        "config": {
            "threshold": threshold,
            "mode": mode,
            "k": k,
            "generator": generator.name if generator else "extractive-v1",
            "index": index.metadata,
        },
        "metrics": metrics,
        "cases": rows,
        "limitations": [
            "Small hand-authored benchmark; not an independently held-out production estimate.",
            "Citation validity measures exact provenance, not semantic entailment.",
            "Gold evidence matching is a conservative substring proxy, not an answer correctness judge.",
            "Provider errors are recorded separately and never counted as successful abstentions.",
        ],
    }


def markdown_report(report: dict) -> str:
    lines = [
        "# Evaluation report",
        "",
        f"Dataset: `{report['dataset']}`",
        "",
        f"Configuration: `{report['config']['generator']}` / `{report['config']['mode']}` / "
        f"threshold `{report['config']['threshold']}`",
        "",
        "| Metric | Measured value |",
        "|---|---:|",
    ]
    lines.extend(f"| {k} | {v if v is not None else 'n/a'} |" for k, v in report["metrics"].items())
    lines += [
        "",
        "## Case results",
        "",
        "| Case | Expected | Result | Retrieval hit |",
        "|---|---|---|---|",
    ]
    for row in report["cases"]:
        lines.append(
            f"| {row['id']} | {'answer' if row['expected_answerable'] else 'abstain'} | "
            f"{row['status']} | {row.get('retrieval_hit', 'n/a')} |"
        )
    lines += ["", "## Limits", ""] + [f"- {item}" for item in report["limitations"]]
    return "\n".join(lines) + "\n"
