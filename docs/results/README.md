# Reference Evaluation Results

These are sanitized reference outputs from the controlled Project Aurora evaluation. They allow readers to inspect the benchmark without requiring Azure model access.

- [`evaluation-summary.md`](evaluation-summary.md): narrative methodology, headline/category metrics, question outcomes, and limitations.
- [`reference-evaluation.csv`](reference-evaluation.csv): one row per pipeline/question case for spreadsheet analysis.
- [`reference-evaluation.json`](reference-evaluation.json): structured configuration, metric definitions, summaries, and case data.

The reference results preserve measured answers, retrieval ranks, scores, timing, and token counts while excluding historical database/run IDs, vector identifiers, collection fingerprints, endpoints, local paths, and full retrieved evidence. Rerunning the benchmark can produce different model prose and timings even with the same controlled setup.
