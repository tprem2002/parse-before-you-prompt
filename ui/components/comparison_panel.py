"""Shared explanatory panels for the parsing comparison."""

from __future__ import annotations

import streamlit as st


def representation_header(title: str, subtitle: str, *, accent: str) -> None:
    color = {"baseline": "#c87800", "docling": "#7857d9", "source": "#00a9c7"}.get(
        accent, "#00a9c7"
    )
    st.markdown(
        f"""
        <div style="border-top:3px solid {color}; padding-top:.65rem; margin-bottom:.75rem">
          <div style="font-weight:750;font-size:1.05rem">{title}</div>
          <div style="opacity:.72;font-size:.8rem">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def baseline_limitations() -> None:
    st.caption(
        "Plain PyMuPDF text · no OCR · no reconstructed table schema · "
        "no exact bounding-box provenance"
    )
