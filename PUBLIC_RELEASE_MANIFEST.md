# Public Release Manifest

## Purpose

This repository is the public companion project for “Parse Before You Prompt: Why Document Intelligence Is the Hidden Layer of Reliable RAG.” It was prepared from the completed `parse-before-you-prompt-demo` source project on 2026-09-02 using an allowlist-copy process.

## Included

- Runnable FastAPI application and Streamlit UI
- PostgreSQL migrations and local PostgreSQL/Chroma Compose configuration
- Public operating scripts
- Synthetic Project Aurora PDF and deterministic ground truth
- Architecture, walkthrough, methodology, and limitations documentation
- Sanitized reference evaluation in Markdown, CSV, and JSON
- Four selected application screenshots and four measured charts
- Final HTML and Markdown article sources

## Excluded

- Source-control metadata, local environment files, credentials, and private cloud configuration
- Model weights and model caches
- PostgreSQL, Chroma, upload, artifact, overlay, and evaluation runtime data
- AI/editor instructions, development prompt reports, diagnostics, and troubleshooting history
- Local paths and historical runtime identifiers not needed for reproducibility

## Reproducibility checks

- Project Aurora PDF SHA-256: `1425eaee4597c0013b1c4933189e57ecfa14557ad0ac086ef06c93c39e412c34`
- Ground-truth SHA-256: `ee0c7929133130e9d35cd046b1550df58de39d2272630b400f38874353e4c447`
- Sanitized result cases: 28

No model weights, `.env`, runtime database/vector data, private Azure endpoint, or personal information are intentionally included.
