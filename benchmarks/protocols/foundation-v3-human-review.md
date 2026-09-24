# Foundation V3 Human Review Decision

**Run:** `61707aba-5ab2-4c16-8ec3-eed74555d69c`
**Decision:** **APPROVED FOR DOWNSTREAM ROUTING ANALYSIS WITH DOCUMENTED LIMITATIONS**

This is a human-review decision. It does not retroactively change the automated result: Criteria 1, 2, and 4 passed, while Criterion 3 failed because Gemini produced exactly three configuration-induced empty/length failures in classification. The frozen automated decision remains **FOUNDATION V3 NOT ACCEPTED — HUMAN REVIEW REQUIRED**.

## Rationale

- 216/224 rows have valid quality labels; 8/224 (3.57%) are missing, compared with 45/224 (20.09%) in V2.
- Configuration-induced empty/length failures fell from 45 to 7.
- 49/56 requests have complete four-model labels.
- Every candidate meets the predeclared minimum: Nemotron 56, Luna 55, Gemini 51, and Sonnet 54 valid labels.
- 39/56 tasks have cross-model acceptable/unacceptable disagreement.
- The cheapest acceptable candidate was Nemotron for 33 tasks, Luna for 10, Sonnet for 5, and Gemini for 0; eight tasks had no acceptable candidate.
- Criterion 3 remains failed because Gemini had exactly three classification configuration failures, meeting the predeclared rejection threshold.

Gemini remains part of the frozen experiment: it has 51 valid labels, including 9 acceptable results, materially different category-level performance, and useful routing-analysis signal. Its remaining failures are retained as reliability/configuration data. Poor quality scores are experimental data, not a benchmark defect.

## Downstream data policy

Rows with valid binary quality labels may be used for supervised quality-model training and evaluation. Missing labels must remain missing and must never be converted to `acceptable = 0`. Preserve their provider outcome, error category, configuration, and available usage, cost, and latency data for separate reliability analysis.

The 224 rows represent **56 unique requests × 4 candidates**, not 224 independent requests. All later train/validation/test splits must be grouped by request/task identity so rows from one request cannot cross split boundaries. Difficulty remains analysis-only metadata and must not become a request-visible router feature.

Foundation V3 is a small supervised dataset. Its 56 requests support routing baselines, oracle analysis, initial router experiments, and a demonstration of the RouteLLM methodology. They do not, by themselves, support broad generalization claims about arbitrary production traffic.

Foundation V3 remains the frozen routing dataset for the next project stages. No Foundation V4 is planned based on model scores. Phase 5.5C is complete following human review; the next phase is **Phase 6 — Routing Baselines + Oracle**. This human decision does not mean Criterion 3 passed.
