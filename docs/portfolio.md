# Portfolio demonstration and GitHub publishing

## Repository positioning

Suggested name: **doccite-rag**

Suggested description:

> Documentation RAG with hybrid retrieval, exact passage citations, and reproducible evaluations of unanswerable questions.

Suggested topics: `rag`, `retrieval-augmented-generation`, `embeddings`, `hybrid-search`,
`llm-evaluation`, `python`, `fastapi`, `ai-engineering`.

This project demonstrates an end-to-end information-retrieval system with explicit
quality boundaries. The interesting conversation is how source provenance,
retrieval, evidence selection, and abstention interact.

## A three-minute demo

1. **Answer a factual question.** Run
   `doccite ask "What is the default timeout for network inactivity?"`.
   Open the citation and point out the pinned commit, exact lines, and source quote.
2. **Inspect the machinery.** Run
   `doccite search "What are the four types of timeouts?"`.
   Explain lexical versus vector retrieval and why rank fusion avoids treating
   incomparable raw scores as if they shared a scale.
3. **Show uncertainty.** Run
   `doccite ask "What uptime SLA does HTTPX guarantee?" --json`.
   Show that weak evidence stops generation and produces no invented citations.
4. **Show measured tradeoffs.** Open `reports/baseline-test.md`. Discuss both
   successful abstention and false abstention; avoid calling the result “100% accurate.”
5. **Show engineering depth.** Open a test rejecting a fabricated quote and explain
   why provider failures are not counted as safe abstentions.

For a richer demo, build the semantic index, select a paid generation model, and
show synthesis from multiple passages. Label that configuration separately and
run its evaluation; the offline baseline numbers do not transfer to it.

## Resume / portfolio copy

> Built a documentation assistant with version-pinned ingestion, local embeddings,
> BM25/vector hybrid retrieval, optional structured LLM generation, and verifiable
> passage citations. Added automated tests and a 44-question authored evaluation
> suite to measure retrieval, citation provenance, and abstention separately.

Only add measured numbers with their denominator, configuration, and dataset scope.
Do not describe exact-match citations as a proof that every claim is true.

## Publish as a separate repository

Copy this folder into a new directory outside any existing repository, including
the `.github` directory. Exclude `.venv`, `.doccite`, caches, and generated build
directories. A source ZIP may also be extracted into the new directory.

Create an empty GitHub repository through the GitHub interface, then run these
commands from the extracted project root, substituting your real remote URL:

```bash
git init -b main
git add .
git commit -m "Build documentation assistant with citations and abstention evaluation"
git remote add origin https://github.com/YOUR-USERNAME/doccite-rag.git
git push -u origin main
```

The workflow is already under `.github/workflows/ci.yml`. It becomes active when
this directory is the repository root. If keeping the project in a monorepo,
move/adapt the workflow to the monorepo's top-level `.github/workflows` and set
the working directory explicitly.

Before publishing, verify `git status`, confirm that no `.env` or credentials
are staged, run the tests, and preserve the upstream license and manifest.
