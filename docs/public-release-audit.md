# Public Release Audit

## Status

**READY FOR USER REVIEW**

The package passed build, result, link, size, and two independent security scans. It is not ready to publish until the repository owner selects a license.

## Included

- Complete FastAPI application code and PostgreSQL migrations
- Complete Streamlit UI using FastAPI as its backend boundary
- Synthetic Project Aurora PDF and deterministic ground truth
- Public architecture, methodology, walkthrough, and limitations documents
- Sanitized evaluation summary plus 28-case CSV and JSON
- Four validated UI screenshots and four measured charts
- Latest available final HTML and Markdown article source
- Public startup, indexing, query, evaluation, reset, model-prefetch, and connection-check scripts

No explicitly versioned v2.1 article existed in the source; the latest `parse-before-you-prompt-final` pair was used. No distinct citation-overlay screenshot was available, so no substitute image was fabricated.

## Excluded

- `.env`, secrets, credentials, and real Azure endpoint/configuration values
- Personal information and local absolute paths
- Model weights and model caches
- PostgreSQL, Chroma, upload, artifact, overlay, and evaluation runtime data
- AI-agent/editor instructions and internal prompt reports
- Development diagnostics, smoke scripts, troubleshooting artifacts, and historical run identifiers
- `.git`

## Security scan

- `scripts/verify_public_release.py`: pass, zero findings
- Independent packaging scan: pass, zero findings
- Personal identifier scan: pass
- Azure/private-environment scan: pass
- Secret/private-key/bearer-token scan: pass
- Private-package-feed scan: pass
- Absolute/local-path scan: pass
- No `.git`, `.env`, `.venv`, runtime `data/`, model cache, database store, or vector store present

## Evaluation

- 14 questions: 12 answerable and two unsupported
- 28 pipeline/question cases with complete baseline/Docling pairing
- Baseline: Recall@1 41.7%, Recall@3 91.7%, Recall@5 91.7%, MRR 0.667, answer match 83.3%, table accuracy 75.0%, precise provenance 0%, unsupported abstention 100%
- Docling standard: Recall@1 75.0%, Recall@3 91.7%, Recall@5 100.0%, MRR 0.836, answer match 100%, table accuracy 100%, precise provenance 100%, unsupported abstention 100%
- Counter-results preserved: baseline table-category MRR 0.750 versus 0.508; baseline broad page/range citation score 100% versus Docling strict citation-page score 91.7%
- Public CSV/JSON headline metrics were recomputed from their case records and matched the completed source evaluation

## Build validation

- `uv lock --check`: pass
- `uv sync --locked --dry-run`: pass; dependencies resolved without installation or download
- Python 3.12.10 `compileall` for `app`, `ui`, and `scripts`: pass using a temporary bytecode directory
- FastAPI import and OpenAPI generation: pass; application imported from this destination and exposed 19 paths
- Streamlit package and public UI support-module imports: pass
- PowerShell script parser: pass
- `docker-compose config --quiet`: pass; the host Docker client emitted a non-project config access warning
- `alembic current`: not run because local services were not required or assumed available
- No automated tests or pytest were run

## File/size audit

- Final file count: 131
- Final size: 3,644,113 bytes (3.475 MiB)
- No file exceeds 20 MiB; no file exceeds 50 MiB
- No model-weight extension was found
- Largest files:
  1. `uv.lock` — 1,042,344 bytes
  2. `demo/source/project_aurora_mission_readiness_report.pdf` — 869,315 bytes
  3. `docs/screenshots/evaluation.png` — 260,716 bytes
  4. `docs/screenshots/parsing-comparison-page-8.png` — 163,735 bytes
  5. `docs/screenshots/structure-and-chunks.png` — 102,542 bytes
  6. `docs/screenshots/ask-and-verify.png` — 89,948 bytes
  7. `docs/blog/assets/evaluation-category-breakdown.png` — 80,176 bytes
  8. `docs/blog/assets/evaluation-answer-metrics.png` — 79,386 bytes
  9. `docs/blog/assets/evaluation-retrieval-metrics.png` — 77,462 bytes
  10. `docs/results/reference-evaluation.json` — 76,990 bytes

## License

No approved source-project license was available. `LICENSE_PENDING.md` exists. Selecting and adding the final license is a **PUBLICATION BLOCKER**.

## Known public-release TODOs

- Choose and add the final license.
- Verify the GitHub organization/repository URL before publishing.
- Perform final editorial review of the article and README.
- Perform a final human inspection of screenshots in the intended GitHub theme.
- Optionally capture a dedicated citation-overlay screenshot; none was available in the completed source package.
