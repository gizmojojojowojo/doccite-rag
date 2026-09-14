# Evaluation report

Dataset: `test.jsonl`

Configuration: `extractive-v1` / `hybrid` / threshold `0.62`

| Metric | Measured value |
|---|---:|
| cases | 32 |
| answerable_cases | 16 |
| unanswerable_cases | 16 |
| errors | 0 |
| retrieval_hit_rate_at_k | 1.0 |
| mrr_at_k | 0.9062 |
| answerable_answer_rate | 0.75 |
| abstention_recall | 1.0 |
| abstention_precision | 0.8 |
| false_answer_rate | 0.0 |
| answer_coverage | 0.375 |
| gold_evidence_answer_precision | 0.9167 |
| citation_validity | 1.0 |
| gold_quote_citation_precision | 0.9167 |
| latency_p50_ms | 2.899 |

## Case results

| Case | Expected | Result | Retrieval hit |
|---|---|---|---|
| test-01 | answer | answered | True |
| test-02 | answer | abstained | True |
| test-03 | answer | answered | True |
| test-04 | answer | answered | True |
| test-05 | answer | answered | True |
| test-06 | answer | answered | True |
| test-07 | answer | abstained | True |
| test-08 | answer | answered | True |
| test-09 | answer | answered | True |
| test-10 | answer | answered | True |
| test-11 | answer | answered | True |
| test-12 | answer | abstained | True |
| test-13 | answer | answered | True |
| test-14 | answer | answered | True |
| test-15 | answer | abstained | True |
| test-16 | answer | answered | True |
| test-17 | abstain | abstained | None |
| test-18 | abstain | abstained | None |
| test-19 | abstain | abstained | None |
| test-20 | abstain | abstained | None |
| test-21 | abstain | abstained | None |
| test-22 | abstain | abstained | None |
| test-23 | abstain | abstained | None |
| test-24 | abstain | abstained | None |
| test-25 | abstain | abstained | None |
| test-26 | abstain | abstained | None |
| test-27 | abstain | abstained | None |
| test-28 | abstain | abstained | None |
| test-29 | abstain | abstained | None |
| test-30 | abstain | abstained | None |
| test-31 | abstain | abstained | None |
| test-32 | abstain | abstained | None |

## Limits

- Small hand-authored benchmark; not an independently held-out production estimate.
- Citation validity measures exact provenance, not semantic entailment.
- Gold evidence matching is a conservative substring proxy, not an answer correctness judge.
- Provider errors are recorded separately and never counted as successful abstentions.
