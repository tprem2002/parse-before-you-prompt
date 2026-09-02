# Controlled Evaluation Methodology

## Design

The benchmark uses one deterministic, synthetic ten-page PDF and two ingestion representations. Both representations use the same Azure `text-embedding-3-large` deployment, 3,072 dimensions, Chroma cosine distance, `top_k=5`, GPT-5.1 deployment, RAG prompt, response schema, citation validator, and abstention contract. Only document representation is intended to differ.

- Baseline: PyMuPDF native text, fixed 800-token windows, 100-token overlap.
- Docling standard: local layout, OCR, tables, pictures, code/formula enrichment, `DoclingDocument`, `HierarchicalChunker` for inspection, and `HybridChunker` for indexing.

The evaluation contains 14 questions: 12 answerable and two unsupported. Four are table questions. Each question ran once against each existing index, producing 28 cases. No LLM judge, reranking, keyword boosting, selective reruns, manual evidence injection, or new indexing was used.

## Relevance and scoring

A retrieval hit is relevant only when its stored source/context text contains all normalized expected terms and its stored page set intersects the expected pages. Page overlap alone does not pass. Unsupported questions are excluded from retrieval and answer-match denominators.

Answer matching uses deterministic Unicode normalization, case folding, punctuation/percent normalization, and complete accepted alternatives. It does not use embeddings or an LLM judge.

## Metrics

- Recall@1, Recall@3, Recall@5: fraction of answerable questions with a strictly relevant hit within the first 1, 3, or 5 results.
- MRR: mean reciprocal rank of the first strictly relevant hit across answerable questions; a miss contributes zero.
- Normalized answer match: fraction of answerable generated answers matching a complete accepted answer after deterministic normalization.
- Table accuracy: normalized answer match restricted to table questions.
- Citation integrity: fraction of supported answers whose claim citation IDs are structurally valid and resolve to retrieved evidence.
- Citation-page accuracy: fraction of supported answers whose application-resolved claim pages satisfy the expected-page rule.
- Precise provenance: fraction of supported answers with stored bounding-box provenance for cited evidence.
- Unsupported abstention: fraction of unsupported questions returning a nonempty insufficient-evidence explanation, no claims, and `insufficient_evidence=true`.
- Answerability decision: correct abstention/non-abstention decision across all questions.
- Latency: persisted retrieval, generation, and total elapsed milliseconds; summaries report mean, median, and p95.

## Interpretation caveat

Baseline citations cover broad page ranges because fixed chunks span multiple pages. Its 100% citation-page score therefore measures page-range intersection, not pinpoint provenance. Docling is judged against stricter per-claim pages backed by exact regions; one correct table answer failed that strict page rule.

Source and ground-truth hashes, case-level results, metric numerators/denominators, latency, and token measurements are retained in the sanitized reference output. Model prose and timing can differ on a rerun even when the controlled configuration is unchanged.
