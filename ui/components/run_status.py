"""Compact processing lifecycle presentation."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import streamlit as st

from components.branding import badge, status_tone


def _elapsed_seconds(run: dict[str, Any]) -> int | None:
    started = run.get("started_at") or run.get("queued_at")
    if not started:
        return None
    try:
        start = datetime.fromisoformat(str(started).replace("Z", "+00:00"))
        end_value = run.get("completed_at")
        end = (
            datetime.fromisoformat(str(end_value).replace("Z", "+00:00"))
            if end_value
            else datetime.now(timezone.utc)
        )
        return max(0, round((end - start).total_seconds()))
    except ValueError:
        return None


def render_run_status(run: dict[str, Any], *, title: str | None = None) -> None:
    status = str(run.get("status") or "unknown")
    label = title or str(run.get("pipeline_type") or "Processing run").replace("_", " ").title()
    with st.container(border=True):
        left, right = st.columns([3, 1])
        left.markdown(f"**{label}**")
        right.markdown(badge(status, status_tone(status)), unsafe_allow_html=True)
        progress = max(0, min(100, int(run.get("progress_percent") or 0)))
        st.progress(progress, text=f"{str(run.get('stage') or status).replace('_', ' ').title()} · {progress}%")
        metadata = st.columns(4)
        metadata[0].caption(f"Run `{str(run.get('id', ''))[:8]}…`")
        elapsed = _elapsed_seconds(run)
        metadata[1].caption(f"Elapsed {elapsed}s" if elapsed is not None else "Elapsed —")
        metadata[2].caption(f"Indexed {'yes' if run.get('indexed') else 'no'}")
        metadata[3].caption(f"Reused {'yes' if run.get('reused') else 'no'}")
        for warning in run.get("warnings") or []:
            st.warning(str(warning), icon="⚠️")
        error = run.get("error")
        if isinstance(error, dict):
            st.error(str(error.get("message") or "Processing failed."))
