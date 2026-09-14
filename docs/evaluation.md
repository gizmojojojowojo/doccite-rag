# Evaluation protocol

## Data

- `benchmarks/dev.jsonl`: 12 cases, six answerable and six unanswerable.
- `benchmarks/test.jsonl`: 32 separate cases, 16 answerable and 16 unanswerable.
- Both are hand-authored against the same pinned 12-document HTTPX subset.
- Positive labels contain exact gold quotes and source paths. Gold items are
  acceptable evidence alternatives, not a requirement to retrieve every item.
- Negatives cover missing context, unsupported guarantees, future versions,
  plausible unsupported features, unrelated tools, a partially answerable
  question, and an adversarial instruction.

“Unanswerable” means the selected snapshot does not establish the requested
answer. It does not mean HTTPX categorically lacks a feature. Source text may
itself be incomplete, ambiguous, or wrong; the system demonstrates traceability,
not independent verification of upstream claims.

The same project author created the implementation and datasets. These are
development and regression fixtures, **not a blind independent test set**. No
model was trained on them. Repeated manual inspection would invalidate a claim
of held-out performance. Add independently authored paraphrases, hard negatives,
multi-passage questions, and larger samples before making general claims.

## Definitions

| Metric | Definition |
|---|---|
| Retrieval hit rate@k | Answerable questions with at least one gold-containing chunk among the first k retrieved chunks / answerable questions |
| MRR@k | Mean reciprocal rank of the first gold-containing chunk; zero for a miss |
| Answerable answer rate | Answerable questions receiving an answer / answerable questions; does not imply the answer is correct |
| Abstention recall | Unanswerable questions receiving an abstention / unanswerable questions |
| Abstention precision | Correct abstentions / all abstentions |
| False-answer rate | Unanswerable questions receiving an answer / unanswerable questions |
| Answer coverage | Answered questions / all questions |
| Gold-evidence answer precision | Answered questions that are labeled answerable and include at least one cited gold quote / answered questions |
| Citation validity | Citations whose quote exists verbatim in the referenced retrieved chunk / emitted citations |
| Gold-quote citation precision | Citations containing an annotated gold quote at the correct source path / emitted citations |
| Errors | Provider failures or invalid model output, reported separately |
| p50 latency | Median successful query duration after index/model loading; excludes startup, download, ingestion, and failed calls |

Undefined ratios are `null` in JSON and `n/a` in Markdown, never a fabricated
100%. Gold quote matching is deliberately strict and can penalize valid alternative
wording or a shorter supporting quote. Gold-evidence precision is a proxy; it
does not verify every clause of a generated answer. Citation validity is expected
to be high because invalid citations are rejected before an answer is emitted.

Provider errors do not count as abstentions. They still remain in the relevant
denominators and cause the evaluation CLI to return a failure exit code.

## Development choices

The initial threshold is 0.62. A development-only sweep of 0.4, 0.5, 0.6, 0.62,
0.7, and 0.8 found several thresholds tied on the small set; 0.62 was retained
without optimizing against the test questions. Use:

```bash
python scripts/compare.py
```

The saved comparison includes corpus and dataset fingerprints, the embedding
backend, mode, threshold, and metrics for every run. Changing embeddings does
not automatically recalibrate the lexical coverage gate; this makes its
limitations visible in the neural comparison too.

## Reproduce the baseline

```bash
doccite ingest
doccite evaluate --dataset benchmarks/test.jsonl \
  --output reports/reproduced.json \
  --max-false-answer-rate 0.10 --min-retrieval-hit-rate 0.85
```

The quality gates are regression checks for this particular authored benchmark,
not production service objectives. The CLI exits 1 if a gate fails or provider
errors occur, and 2 for invalid configuration or input. A normal abstention is
successful command execution, exit 0.

## Recorded baseline interpretation

The recorded run retrieves gold evidence for all 16 positive questions. It answers
12, of which 11 quote labeled evidence; it abstains on all 16 negative questions
and on four positive questions. The negative result of zero false answers in 16
examples has substantial sampling uncertainty. It does not establish that
hallucination has been solved.

The four false abstentions and one evidence-selection miss are visible in the
JSON report. They are useful engineering leads: distinguish search misses from
an overly conservative gate and from a poor answer-selection step. Do not fix
these by adding question-specific branches or hardcoding benchmark answers.

## Next evaluation work

1. Have another person annotate a fresh test set and adjudicate ambiguous labels.
2. Add larger hard-negative and paraphrase sets with source evidence rationale.
3. Compare neural retrieval and reranking while holding questions fixed.
4. Run paid generation only with an explicitly chosen model, report costs and
   provider errors, and review every generated claim for entailment.
5. Calibrate an answerability classifier on development labels and plot coverage
   against false-answer risk, with confidence intervals on a sufficiently large set.
