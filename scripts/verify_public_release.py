from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path


MAX_FILE_BYTES = 50 * 1024 * 1024
BINARY_SUFFIXES = {
    ".bmp",
    ".gif",
    ".ico",
    ".jpeg",
    ".jpg",
    ".pdf",
    ".png",
    ".webp",
    ".zip",
}
SKIP_DIRS = {
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "logs",
    "temp",
    "tmp",
    "venv",
}
FORBIDDEN_DIRS = {
    ".git": "git_metadata",
    ".chroma": "vector_runtime",
    "chroma": "vector_runtime",
    "data": "runtime_data",
    "huggingface": "model_cache",
    "model-cache": "model_cache",
    "pgdata": "database_runtime",
    "postgres-data": "database_runtime",
    "postgres_data": "database_runtime",
}
PLACEHOLDER_MARKERS = (
    "YOUR_",
    "<YOUR-",
    "<YOUR_",
    "REPLACE_ME",
    "EXAMPLE",
)


@dataclass(frozen=True, order=True)
class Finding:
    path: str
    category: str
    line: int | None = None


PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "api_key_assignment",
        re.compile(
            r"(?m)^(?:AZURE_OPENAI_API_KEY|OPENAI_API_KEY|API_KEY)=([^\r\n#]*)$"
        ),
    ),
    ("bearer_token", re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{16,}")),
    ("openai_secret_key", re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b")),
    (
        "private_key",
        re.compile("BEGIN " + r"(?:RSA |EC |OPENSSH )?PRIVATE KEY"),
    ),
    (
        "database_url_password",
        re.compile(
            r"(?i)\b(?:postgres(?:ql)?(?:\+psycopg)?|mysql|mariadb)://"
            r"[^:/\s]+:([^@/\s]+)@([^/\s]+)"
        ),
    ),
    (
        "azure_openai_endpoint",
        re.compile(
            r"(?i)https://[A-Za-z0-9-]+\.(?:openai|cognitiveservices)\.azure\.(?:com|us)(?:/[^\s\"'<>)]*)?"
        ),
    ),
    ("windows_home_path", re.compile(r"(?i)\b[A-Z]:\\Users\\[^\\\s]+")),
    (
        "macos_home_path",
        re.compile(r"(?i)(?<![A-Za-z0-9_])/" + r"Users/[^/\s]+"),
    ),
    (
        "linux_home_path",
        re.compile(r"(?i)(?<![A-Za-z0-9_])/" + r"home/[^/\s]+"),
    ),
    ("file_url", re.compile(r"(?i)\bfile://[^\s\"'<>]+")),
    (
        "private_package_registry",
        re.compile(
            r"(?i)https?://[^\s\"'<>]*(?:pkgs\.dev\.azure\.com|artifactory|nexus|private[-.]pypi)[^\s\"'<>]*"
        ),
    ),
    (
        "credential_in_url",
        re.compile(r"(?i)https?://[^/@\s:]+:[^/@\s]+@[^/\s]+"),
    ),
    (
        "absolute_artifact_path",
        re.compile(
            r"(?i)(?:[A-Z]:\\|/(?:Users|home|mnt|opt|srv|var)/)[^\r\n]*[\\/]data[\\/]artifacts[\\/]"
        ),
    ),
    (
        "cloud_account_identifier",
        re.compile(
            r"(?im)^\s*(?:AZURE_(?:TENANT|SUBSCRIPTION|CLIENT)_ID|TENANT_ID|SUBSCRIPTION_ID|ACCOUNT_ID)\s*[=:]\s*([0-9a-f-]{32,36})\s*$"
        ),
    ),
)


def is_placeholder(value: str) -> bool:
    upper = value.upper()
    return not value.strip() or any(marker in upper for marker in PLACEHOLDER_MARKERS)


def ignored_match(category: str, match: re.Match[str]) -> bool:
    value = match.group(0)
    if category == "api_key_assignment":
        return is_placeholder(match.group(1))
    if category == "azure_openai_endpoint":
        return is_placeholder(value)
    if category == "database_url_password":
        password, host = match.group(1), match.group(2).split(":", 1)[0].lower()
        if is_placeholder(password):
            return True
        return password == "parse_prompt" and host in {"localhost", "127.0.0.1"}
    if category == "cloud_account_identifier":
        return is_placeholder(match.group(1))
    return False


def scan(root: Path) -> list[Finding]:
    findings: set[Finding] = set()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        relative_text = relative.as_posix()

        if path.is_dir():
            category = FORBIDDEN_DIRS.get(path.name.lower())
            if category:
                findings.add(Finding(relative_text, category))
            continue

        if any(part.lower() in SKIP_DIRS or part.lower() in FORBIDDEN_DIRS for part in relative.parts[:-1]):
            continue
        if path.name == ".env" or (path.name.startswith(".env.") and path.name != ".env.example"):
            findings.add(Finding(relative_text, "environment_file"))
        if path.stat().st_size > MAX_FILE_BYTES:
            findings.add(Finding(relative_text, "file_over_50_mb"))
        if path.suffix.lower() in BINARY_SUFFIXES:
            continue

        raw = path.read_bytes()
        if b"\x00" in raw[:8192]:
            continue
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            findings.add(Finding(relative_text, "non_utf8_text_candidate"))
            continue

        for category, pattern in PATTERNS:
            for match in pattern.finditer(text):
                if ignored_match(category, match):
                    continue
                line = text.count("\n", 0, match.start()) + 1
                findings.add(Finding(relative_text, category, line))

    return sorted(findings)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    findings = scan(root)
    if findings:
        for finding in findings:
            suffix = f" | line {finding.line}" if finding.line is not None else ""
            print(f"{finding.path} | {finding.category}{suffix}")
        return 1
    print(". | scan_pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
