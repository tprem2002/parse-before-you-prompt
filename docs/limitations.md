# Limitations

This repository is a focused engineering demonstration, not a universal parser ranking.

- The benchmark uses one synthetic ten-page report, 14 questions, and one evaluation pass.
- Only one embedding/chat configuration was measured; model prose and latency can vary between runs.
- Docling standard conversion was expensive on CPU: about 32.6 minutes for this document and quality profile.
- Recovered hierarchy was useful but imperfect, and some vertical table-header merges remained imperfect.
- Picture descriptions were generic, derived retrieval aids rather than verbatim source evidence.
- Table-header repetition was configured but not exercised because production chunks fit below 800 tokens.
- The baseline produced only three broad chunks. Those chunks can rank strongly, and Recall@5 is a weak discriminator.
- Recall@3 tied at 91.7%.
- Baseline table-category MRR (0.750) exceeded Docling (0.508), despite lower baseline table-answer accuracy.
- Baseline broad page/range citation scoring reached 100%; Docling's strict citation-page score was 91.7% because one correct answer's claims did not all resolve to an expected page.
- Citation integrity proves that cited IDs exist and are structurally valid; it does not prove semantic entailment.
- Precise provenance makes verification possible but does not prevent hallucination.
- Granite-Docling-258M remains an optional comparison path and was not evaluated.

The measured conclusion is narrow: on Project Aurora, preserving OCR, table associations, and exact source regions improved several important retrieval, answer, provenance, and context-efficiency outcomes, while imposing substantially higher preprocessing cost.
