# Project Aurora Reference Evaluation

## Scope and identity

This sanitized reference describes one completed controlled evaluation of the synthetic ten-page Project Aurora report. It is descriptive, not statistically significant or a universal parser ranking.

- Evaluation version: `parse-before-you-prompt-evaluation-v1`
- Metric definition version: `parse-before-you-prompt-metrics-v1`
- Source PDF SHA-256: `1425eaee4597c0013b1c4933189e57ecfa14557ad0ac086ef06c93c39e412c34`
- Ground-truth SHA-256: `ee0c7929133130e9d35cd046b1550df58de39d2272630b400f38874353e4c447`
- Questions: 14 (12 answerable, two unsupported; four table questions)
- Pipeline/question cases: 28, with no missing, duplicate, provider-failed, or execution-failed cases

Both paths used `text-embedding-3-large`, 3,072-dimensional externally generated vectors, Chroma cosine distance, top-k 5, GPT-5.1, low reasoning effort, prompt `parse-before-you-prompt-rag-v1`, and response schema `parse-before-you-prompt-answer-v1`. The baseline index contained three chunks; the Docling standard index contained 25 Hybrid chunks.

## Headline metrics

| Metric | Baseline | Docling standard | Difference |
|---|---:|---:|---:|
| Recall@1 | 41.7% (5/12) | 75.0% (9/12) | +33.3 pp |
| Recall@3 | 91.7% (11/12) | 91.7% (11/12) | 0.0 pp |
| Recall@5 | 91.7% (11/12) | 100.0% (12/12) | +8.3 pp |
| MRR | 0.667 | 0.836 | +0.169 |
| Normalized answer match | 83.3% (10/12) | 100.0% (12/12) | +16.7 pp |
| Table-question accuracy | 75.0% (3/4) | 100.0% (4/4) | +25.0 pp |
| Citation integrity | 100.0% (11/11) | 100.0% (12/12) | 0.0 pp |
| Citation-page accuracy | 100.0% (11/11) | 91.7% (11/12) | −8.3 pp |
| Precise provenance | 0.0% (0/11) | 100.0% (12/12) | +100.0 pp |
| Unsupported abstention | 100.0% (2/2) | 100.0% (2/2) | 0.0 pp |
| Answerability decision | 92.9% (13/14) | 100.0% (14/14) | +7.1 pp |
| Mean total latency | 3,178.50 ms | 2,959.57 ms | −218.93 ms |

Recall@5 is a weak discriminator for a baseline with only three broad chunks. Recall@1 and MRR show more of the representation effect.

## Category results

| Kind (n) | Pipeline | R@1 | R@3 | R@5 | MRR | Answer match | Citation page | Mean total ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| code (1) | Baseline | 100% | 100% | 100% | 1.000 | 100% | 100% | 3,261 |
| code (1) | Docling | 100% | 100% | 100% | 1.000 | 100% | 100% | 2,932 |
| list (1) | Baseline | 0% | 100% | 100% | 0.500 | 100% | 100% | 2,439 |
| list (1) | Docling | 100% | 100% | 100% | 1.000 | 100% | 100% | 2,196 |
| picture (2) | Baseline | 50% | 100% | 100% | 0.750 | 100% | 100% | 2,200 |
| picture (2) | Docling | 100% | 100% | 100% | 1.000 | 100% | 100% | 2,369 |
| table (4) | Baseline | 50% | 100% | 100% | 0.750 | 75% | 100% | 4,935 |
| table (4) | Docling | 25% | 75% | 100% | 0.508 | 100% | 75% | 3,803 |
| text (4) | Baseline | 25% | 75% | 75% | 0.500 | 75% | 100% | 2,340 |
| text (4) | Docling | 100% | 100% | 100% | 1.000 | 100% | 100% | 2,943 |
| unsupported (2) | Baseline | N/A | N/A | N/A | N/A | N/A | N/A | 2,652 |
| unsupported (2) | Docling | N/A | N/A | N/A | N/A | N/A | N/A | 2,292 |

## Per-question summary

`rank` is the first strictly relevant retrieval rank; A is normalized answer match, C is citation-page correctness, and P is precise provenance.

| ID | Kind | Baseline rank / A / C / P | Docling rank / A / C / P | Finding |
|---|---|---|---|---|
| Q01 | table | 1 / yes / yes / no | 2 / yes / yes / yes | Both answered correctly. |
| Q02 | table | 1 / no / yes / no | 1 / yes / yes / yes | Baseline missed the complete multi-part association. |
| Q03 | table | 2 / yes / yes / no | 3 / yes / no / yes | Docling answer matched; its claims did not all resolve to expected pages. |
| Q04 | table | 2 / yes / yes / no | 5 / yes / yes / yes | Baseline ranked relevant evidence higher; both answers matched. |
| Q05 | list | 2 / yes / yes / no | 1 / yes / yes / yes | Docling ranked the evidence first. |
| Q06 | picture | 1 / yes / yes / no | 1 / yes / yes / yes | Both matched; only Docling had exact regions. |
| Q07 | picture | 2 / yes / yes / no | 1 / yes / yes / yes | Docling ranked the evidence first. |
| Q08 | text/OCR | miss / no / N/A / N/A | 1 / yes / yes / yes | Baseline lacked the OCR fact; Docling recovered 15 minutes. |
| Q09 | text | 2 / yes / yes / no | 1 / yes / yes / yes | Docling ranked the evidence first. |
| Q10 | text | 2 / yes / yes / no | 1 / yes / yes / yes | Docling ranked the evidence first. |
| Q11 | text | 1 / yes / yes / no | 1 / yes / yes / yes | Both answered correctly. |
| Q12 | code | 1 / yes / yes / no | 1 / yes / yes / yes | Both answered correctly. |
| Q13 | unsupported | N/A | N/A | Both structurally abstained. |
| Q14 | unsupported | N/A | N/A | Both structurally abstained. |

## OCR, tables, and provenance

The baseline's only representation miss was Q08: the recovery-window fact existed only in a scanned appendix and never entered its three-chunk index. It therefore returned no relevant rank and abstained. Local RapidOCR placed the **15 minutes** value into Docling's representation, which retrieved it at rank one with a page-8 bounding box.

Docling retained the important REQ-207 and REQ-209 table associations and answered all four table questions. Its table-category MRR was nevertheless lower than baseline (0.508 versus 0.750), and Q03's correct answer failed the strict citation-page rule. Some vertical table-header merges remained imperfect.

Both paths achieved 100% citation-ID/structure integrity among supported answers. Baseline's broad page ranges produced 100% page intersection for 11 supported answers but no exact regions. All 12 supported Docling answers had precise stored provenance. Citation integrity is not semantic entailment, and provenance does not eliminate hallucination.

## Latency and tokens

| Measure | Baseline | Docling standard |
|---|---:|---:|
| Retrieval mean / median / p95 | 496 / 348 / 2,219 ms | 451 / 436 / 948 ms |
| Generation mean / median / p95 | 2,660 / 2,063 / 5,141 ms | 2,478 / 2,268 / 3,918 ms |
| Total mean / median / p95 | 3,179 / 2,582 / 6,508 ms | 2,960 / 2,762 / 4,198 ms |
| Chat input / output / total tokens | 32,537 / 1,437 / 33,974 | 13,951 / 1,374 / 15,325 |
| Mean evidence tokens/question | 1,913.00 | 495.21 |

Persisted ingestion timings were not rerun: baseline ingestion was 52 ms; Docling conversion/chunking was 1,954,758 ms; and chunk/provenance backfill was 466 ms. These timings describe one machine and quality profile.

## Method and limitations

Strict retrieval relevance required both expected-page intersection and all normalized expected terms. Answers used deterministic normalization and complete accepted alternatives—no LLM judge. No reranking, keyword boosting, selective reruns, or manually injected evidence was used.

The results are limited by the single synthetic document, small question set, one model pass, costly CPU conversion, imperfect hierarchy/table reconstruction, derived picture descriptions, and unexercised header repetition. Granite-Docling-258M was not evaluated. See [methodology](../methodology.md), [limitations](../limitations.md), and the case-level [CSV](reference-evaluation.csv) and [JSON](reference-evaluation.json).
