# Parse Before You Prompt

## Why document intelligence is the hidden layer of reliable RAG

**Prem Tamang · Gen AI Lead · Continuum Resources · August 2026**

### The field note in 60 seconds

| | |
|---|---|
| **Intent** | Test parsing as a RAG quality variable while holding the downstream stack constant. |
| **Built** | Runnable PyMuPDF baseline vs. Docling standard application using FastAPI, PostgreSQL, Chroma, Streamlit, `text-embedding-3-large`, and GPT-5.1. |
| **Finding** | Recall@1 improved from 41.7% to 75%; Docling reached 100% answer match and table-question accuracy. |
| **Why it matters** | Precise provenance availability moved from 0% to 100%; Docling also recovered OCR-only evidence the baseline never indexed. |

> **A retriever cannot recover evidence that ingestion already destroyed.** Better models help only after the evidence survives parsing.

## 01 · Why we ran this

Most RAG optimization starts downstream: a better embedding model, reranking, larger context, stronger prompts, or a newer LLM. Those choices matter, but they all operate on the representation produced during ingestion.

A technical PDF carries more than text: reading order, hierarchy, row/column relationships, captions, diagrams, formulas, code, footnotes, and scanned content. If parsing interleaves columns, flattens a table, drops OCR-only text, or discards source geometry, the retriever is searching a damaged version of the document.

## 02 · Controlled experiment

We created a deterministic ten-page **Project Aurora Mission Readiness Report** containing two-column prose, nested headings, multi-page verification tables, a readiness chart, architecture diagram, rotated scanned appendix, formula, code, footnotes, and retrieval distractors.

| Baseline | Docling standard |
|---|---|
| PyMuPDF native text | Structured `DoclingDocument` |
| 3 fixed windows | Layout + reading order + OCR + table structure |
| 800-token max + 100 overlap | Hierarchical + Hybrid chunking |
| No OCR | RapidOCR |
| Broad page ranges | Page/item/bounding-box provenance |
| No exact overlays | Exact evidence overlays |

Held constant: `text-embedding-3-large` (3072d), Chroma cosine, top-k 5, same prompt/schema/validator, GPT-5.1 with low reasoning effort.

## 03 · What we actually built

The comparison was an end-to-end application:

`Upload → persisted run → local parse → chunks → embeddings → Chroma → GPT-5.1 → validated evidence IDs → page/bbox → highlighted evidence`

- **FastAPI:** document, processing, chunk, RAG, evidence, evaluation APIs
- **PostgreSQL:** authoritative runs, chunks, provenance, retrieval hits, evaluations
- **Chroma:** external vectors and lightweight search metadata
- **Filesystem:** source PDF, lossless Docling JSON, page images, crops, overlays
- **Streamlit:** Parsing Comparison, Structure & Chunks, Ask & Verify, Evaluation

## 04 · What Docling changed before embedding

Our benchmark pinned **Docling 2.123.1** and used a local standard quality pipeline:

- **Heron:** layout and reading order
- **TableFormer Accurate:** table structure
- **RapidOCR:** scanned content
- **Document Figure Classifier v2:** figure classification
- **Granite Vision 3.3 2B:** derived picture descriptions
- **CodeFormulaV2:** formula/code extraction

| Source | Searchable representation | Evidence retained |
|---|---|---|
| Paragraph/list | Text + contextual headings | item, page, bbox, char span |
| Table | structured row/column serialization | table object, crop, refs, regions |
| Picture/chart | caption + labeled derived description | crop, classification, region |
| Formula/code | extracted representation | source item + region |
| OCR | recognized source text | OCR-backed item + bbox |

## 05 · Chunking and provenance

- 47 hierarchical inspection chunks
- 25 Hybrid vector chunks
- 10 peer merges
- 0 production splits
- 364 maximum contextualized tokens under an 800-token ceiling
- 130 provenance regions

**HierarchicalChunker** preserves natural structure for inspection. **HybridChunker** makes those units embedding-friendly with token-aware contextualization.

## 06 · Strongest measured proof

The scanned appendix contained:

> **Maximum recovery window: 15 minutes.**

**Baseline:** PyMuPDF never extracted the sentence, so the correct answer was an abstention.

**Docling:** RapidOCR recovered it; the system answered **15 minutes** and resolved the evidence to page 8, Docling item `#/texts/121`, and stored bounding boxes.

The LLM did not become smarter. The evidence became available.

A second failure mode appeared in **REQ-205**: baseline retrieved a broad relevant chunk but associated the requirement with **Guidance**. The accepted relation was **Navigation**, verified by **Analysis**. Docling preserved that table relationship and answered correctly.

## 07 · Measured RAG impact

| Metric | Baseline | Docling |
|---|---:|---:|
| Recall@1 | 41.7% | **75.0%** |
| Recall@3 | **91.7%** | **91.7%** |
| Recall@5 | 91.7% | **100.0%** |
| MRR | 0.667 | **0.836** |
| Answer match | 83.3% | **100.0%** |
| Table accuracy | 75% | **100%** |
| Precise provenance | 0% | **100%** |
| Unsupported abstention | **100%** | **100%** |

The most useful headline is Recall@1 (+33.3 pp), not Recall@5, because baseline only contained three broad chunks.

## 08 · Grounding and provenance

The application, not GPT-5.1, owned source geometry:

**Claim → Evidence ID → Retrieved chunk → Docling item → Page → Bounding box → Highlighted region**

| Evidence property | Baseline | Docling |
|---|---:|---:|
| Citation-ID integrity | 100% | 100% |
| Page/range accuracy | 100% | 91.7% |
| Exact region provenance | 0% | **100%** |
| Evidence overlay | No | **Yes** |

The one Docling citation-page miss is intentionally retained. Provenance does not eliminate hallucination; it makes citation problems inspectable.

## 09 · Where Docling was stronger, and where baseline held its own

**Docling stronger:** OCR recovery, Recall@1, overall MRR, answer match, table accuracy, precise provenance, smaller evidence context.

**Baseline held its own:** Recall@3 tied, table-category MRR was higher (0.750 vs. 0.508), broad page/range scoring was higher, code question tied, parsing was dramatically faster.

## 10 · Not benchmarked here: where Docling could go next

Current Docling documentation exposes capabilities beyond our pinned experiment:

- **Granite-Docling-258M full-page VLM conversion** producing structured DocTags.
- **Chart extraction** that can convert bar/pie/line charts into structured data such as CSV.
- **VLM table structure recognition**, including Granite Vision 4.1 4B using OTSL.
- **GPU/vLLM runtimes** for higher-throughput VLM conversion.
- **Multiple OCR and vision model families** for different document domains/languages.
- **Broader RAG integrations** across frameworks and search/vector backends.

These are future experiments, not Project Aurora benchmark results.

## 11 · Cost and deployment reality

| Measure | Baseline | Docling |
|---|---:|---:|
| Local processing | **52 ms** | 32.6 min CPU |
| Evidence tokens/query | 1,913 | **495** |
| Chat input tokens | 32,537 | **13,951** |
| Total chat tokens | 33,974 | **15,325** |
| Mean query latency | 3,179 ms | **2,960 ms** |
| p95 latency | 6,508 ms | **4,198 ms** |

The CPU conversion time is descriptive of our quality profile and hardware, not a general Docling throughput claim.

## 12 · Conclusion

Docling did not replace RAG. It improved the **evidence model** RAG received: OCR evidence survived, table relationships remained addressable, chunks were more focused, and citations could resolve to exact source regions.

> **Validate the ingestion layer with the same rigor as the model layer. A bigger LLM cannot retrieve evidence that never entered the index.**
>
> **Parse accurately. Retrieve structurally. Answer with evidence.**

## Companion demo

[https://github.com/tprem2002/parse-before-you-prompt](https://github.com/tprem2002/parse-before-you-prompt)

The repository contains Project Aurora, both ingestion paths, FastAPI, Streamlit, ground truth, reference evaluation outputs, and evidence overlays.

## Sources

- https://docling-project.github.io/docling/
- https://docling-project.github.io/docling/usage/model_catalog/
- https://docling-project.github.io/docling/usage/vision_models/
- https://docling-project.github.io/docling/reference/pipeline_options/
- https://docling-project.github.io/docling/concepts/docling_document/
