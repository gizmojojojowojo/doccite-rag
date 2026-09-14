# Architecture and decisions

## Source provenance is part of the data model

The bundled corpus contains 12 Markdown documents from HTTPX 0.28.1 at commit
`26d48e0634e6ee9cdc0533996db289ce4b430177`. It is a curated subset, not the entire
HTTPX documentation site or a promise of current behavior. The manifest lists
relative paths, source URLs, and SHA-256 checksums. The original BSD license is included.

Ingestion rejects changed bytes, duplicate paths, paths outside the corpus, and
URLs that disagree with the pinned repository and revision. No question can
trigger a crawl, fetch an arbitrary URL, or add a document to the index.

The chunker preserves source text and one-based, inclusive line numbers. It packs
paragraph blocks within a heading section, carries heading context in retrieval
metadata, and keeps fenced code blocks together. The 1,200-character budget is
soft: a long indivisible paragraph or code block may exceed it. No overlap is
added; adjacent chunks can be retrieved separately. This keeps citations simple
but can separate an explanation from a following example.

Chunk IDs hash source digest, path, line span, and chunker version. They remain
stable on a repeated build with identical inputs and settings.

## Explicit retrieval baselines

`HashEmbedder` maps unigrams and bigrams to 512 signed dimensions and L2-normalizes
the result. It provides deterministic vector-search plumbing with zero external
dependencies. It does not understand meaning or synonyms.

`SentenceTransformerEmbedder` uses the pinned MiniLM model through
[Sentence Transformers](https://sbert.net/docs/quickstart.html) and returns normalized
384-dimensional vectors. CPU execution and fixed model revision simplify
reproduction. Its tokenizer can truncate long passages to the model context
limit; full source text remains stored for citation. Improving token-aware
chunking is a clear next experiment, especially for oversized code blocks.

SQLite stores metadata, source chunks, and vectors. Index creation happens in a
temporary database followed by atomic replacement. Failed embedding calls leave
the previous index intact. Readers load the complete index once and close the
database; an API restart is necessary to pick up a new index.

BM25 uses `k1=1.5`, `b=0.75`. Cosine search operates on normalized vectors.
Hybrid search takes up to 50 candidates from each ranking and combines them via
reciprocal rank fusion: `sum(1 / (60 + rank))`. Ties use stable chunk IDs.
The index records its embedding backend and dimensions and rejects mismatches.

This is exhaustive, in-memory search: roughly O(chunks × vector dimensions).
For a 117-passage corpus it is easy to inspect and fast. At larger scale, replace
it with a vector index, persist an inverted lexical index, and compare recall
before adopting approximate nearest-neighbor search.

## Abstention and generation

The gate scores a retrieved passage by the fraction of query-term IDF weight
covered by its title and text. Unseen terms receive a high weight, penalizing
questions about missing guarantees, versions, and context. The default threshold
is 0.62. This heuristic is not trained, semantically calibrated, or a proof of
answerability. It can miss paraphrases, be manipulated by word choice, or accept
partially answerable questions. Measure it on the target corpus.

Only passages passing the gate go to the generator. The extractive baseline
selects a source block; procedural questions can return a full passage so that
code examples are retained. It cannot synthesize a complete multi-part answer.

The optional OpenAI adapter uses the Responses API and a strict JSON schema with
`answerable`, atomic claims, and evidence items. Documentation and questions are
explicitly delimited as untrusted JSON data. The prompt asks the model to abstain
when the whole question is unsupported, but this is mitigation, not a guarantee
against prompt injection or hallucination. There are no tool-execution capabilities.

Application code constructs every citation. The model supplies a retrieved chunk
ID and a verbatim quote; it never supplies the authoritative URL or line numbers.
Validation rejects missing citations, unknown IDs, invented quotes, empty claims,
and contradictory abstention output. Lines are calculated from the quote offset
in the original passage, and URLs are pinned to the source commit.

**Provenance validity does not establish semantic entailment.** A model could cite
an authentic paragraph that does not support its claim. The benchmark separately
checks labeled gold-quote inclusion, but that is still not a semantic correctness
judge. Human claim-level review or an independently validated entailment grader
is necessary before stronger reliability claims.

## Error and operational boundaries

- Weak evidence: ordinary `abstained` response; no generation call.
- Model reports insufficient evidence: `abstained`, no claims or citations.
- Provider timeout, failure, refusal, truncation, malformed output, or invalid
  citation: explicit error; never scored as a correct abstention.
- API input: 1–2,000 nonblank characters, `k` from 1 to 20; model credentials
  and generation backend are server-side settings.
- API capacity: one active question per process, HTTP 429 while occupied;
  generation uses a 60-second network timeout. No automatic paid retries.
- Traces include retrieval evidence and timing but no API keys. Evaluation
  reports intentionally store questions, answers, and quotations locally.

Tests exercise invariants and failure paths. Paid API behavior is tested with
mock responses; recorded offline results must not be presented as LLM performance.
