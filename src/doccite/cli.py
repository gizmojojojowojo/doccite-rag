import argparse
import json
import sys
from pathlib import Path

from doccite.answer import GenerationError, OpenAIGenerator, ask
from doccite.embeddings import make_embedder
from doccite.evaluate import evaluate, markdown_report
from doccite.ingest import DEFAULT_CORPUS, load_corpus
from doccite.search import SearchIndex, build_index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="doccite", description="Documentation answers with exact, verifiable source passages."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest", help="Verify the corpus and atomically build a local index")
    ingest.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ingest.add_argument("--embeddings", choices=["hash", "semantic"], default="hash")
    ingest.add_argument("--max-chars", type=int, default=1200)
    question = sub.add_parser("ask", help="Ask a question and show cited evidence")
    question.add_argument("question")
    question.add_argument(
        "--json", action="store_true", help="Include the complete retrieval trace"
    )
    search = sub.add_parser("search", help="Inspect retrieved passages and their component scores")
    search.add_argument("question")
    evaluation = sub.add_parser("evaluate", help="Run a labeled benchmark and save actual results")
    evaluation.add_argument("--dataset", type=Path, default=Path("benchmarks/test.jsonl"))
    evaluation.add_argument("--output", type=Path, default=Path("reports/latest.json"))
    evaluation.add_argument("--max-false-answer-rate", type=float)
    evaluation.add_argument("--min-retrieval-hit-rate", type=float)
    info = sub.add_parser("info", help="Show corpus revision and index configuration")
    for p in [ingest, question, search, evaluation, info]:
        p.add_argument("--index", type=Path, default=Path(".doccite/index.sqlite"))
    for p in [question, search, evaluation]:
        p.add_argument("--mode", choices=["hybrid", "bm25", "dense"], default="hybrid")
        p.add_argument("-k", type=int, default=5)
    for p in [question, evaluation]:
        p.add_argument(
            "--threshold",
            type=float,
            default=0.62,
            help="IDF-weighted query coverage gate; not a probability",
        )
        p.add_argument("--generator", choices=["extractive", "openai"], default="extractive")
        p.add_argument("--model", help="OpenAI model ID; alternatively set DOCCITE_MODEL")
    args = parser.parse_args(argv)
    try:
        if args.command == "ingest":
            chunks, metadata = load_corpus(args.corpus, max_chars=args.max_chars)
            print(
                json.dumps(
                    build_index(args.index, chunks, metadata, make_embedder(args.embeddings)),
                    indent=2,
                )
            )
            return 0
        index = SearchIndex(args.index)
        if args.command == "info":
            print(json.dumps(index.metadata, indent=2))
            return 0
        if args.command == "search":
            for hit in index.search(args.question, k=args.k, mode=args.mode):
                print(
                    f"\n{hit.chunk.title}\n{hit.chunk.url}\n"
                    f"RRF={hit.score:.4f} BM25={hit.bm25:.3f} cosine={hit.cosine:.3f} "
                    f"coverage={hit.coverage:.3f}\n\n{hit.chunk.text}"
                )
            return 0
        generator = OpenAIGenerator(args.model) if args.generator == "openai" else None
        if args.command == "ask":
            result = ask(
                index,
                args.question,
                generator=generator,
                threshold=args.threshold,
                mode=args.mode,
                k=args.k,
            )
            print(json.dumps(result.to_dict(), indent=2) if args.json else result.markdown())
            return 0
        for bound in [args.max_false_answer_rate, args.min_retrieval_hit_rate]:
            if bound is not None and not 0 <= bound <= 1:
                raise ValueError("Evaluation quality gates must be between 0 and 1")
        report = evaluate(
            index,
            args.dataset,
            generator=generator,
            threshold=args.threshold,
            mode=args.mode,
            k=args.k,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        args.output.with_suffix(".md").write_text(markdown_report(report), encoding="utf-8")
        print(json.dumps(report["metrics"], indent=2))
        print(f"\nSaved {args.output} and {args.output.with_suffix('.md')}")
        metrics = report["metrics"]
        failed = metrics["errors"] > 0
        if args.max_false_answer_rate is not None:
            failed |= (
                metrics["false_answer_rate"] is None
                or metrics["false_answer_rate"] > args.max_false_answer_rate
            )
        if args.min_retrieval_hit_rate is not None:
            failed |= (
                metrics["retrieval_hit_rate_at_k"] is None
                or metrics["retrieval_hit_rate_at_k"] < args.min_retrieval_hit_rate
            )
        return 1 if failed else 0
    except (ValueError, OSError, GenerationError) as exc:
        print(f"doccite: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
