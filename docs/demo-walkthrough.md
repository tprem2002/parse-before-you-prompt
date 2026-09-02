# Five-Minute Demo Walkthrough

This walkthrough can use the live local demo or the repository screenshots and reference results. It does not require rerunning the evaluation.

1. Start FastAPI and Streamlit with `./scripts/start_demo.ps1`.
2. On Home, choose **Use Project Aurora demo**.
3. Open **Parsing Comparison** and jump to page 8.
4. Compare the missing baseline OCR text with the Docling representation of the scanned appendix.
5. Open **Structure and Chunks**. Inspect the table, OCR, and Hybrid chunks; note the page and provenance labels.
6. Open **Ask and Verify** and ask: “What is the maximum recovery window?”
7. Compare the baseline's evidence-based abstention with Docling's **15 minutes** answer.
8. Open the Docling citation and inspect the highlighted page-8 region when runtime overlays are available.
9. Ask the REQ-205 table question from the evaluation. Compare the baseline association error with Docling's preserved Navigation + Analysis relationship. REQ-207 and REQ-209 are additional useful table examples.
10. Open **Evaluation** and show the completed 28-case comparison, including Recall@1, MRR, answer match, provenance, token use, and counter-results.

Without live services, use [the selected screenshots](screenshots/), [evaluation summary](results/evaluation-summary.md), and [reference data](results/README.md).
