# Security

This is a demonstration and research repository, not a production service.

- Never commit `.env`, cloud credentials, tokens, or private keys.
- Use your own Azure OpenAI resource and deployment names.
- Treat demo reset as destructive for project-owned local database, vector, upload, artifact, and evaluation data.
- Do not expose the FastAPI demo directly to the internet without authentication, authorization, network controls, and production hardening.
- Do not treat the one-worker, in-process demo architecture as a production distributed queue.

Report repository security issues through the repository's GitHub issue process unless an organization-level reporting process is supplied later. Do not include secrets in an issue.
