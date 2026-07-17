"""PDF upload gate — a session can't chat until at least one PDF is
processed. Supports uploading and indexing multiple PDFs at once."""

import os
import tempfile

import streamlit as st

from backend.ingestion_pipeline import process_pdf
from backend.session_store import save_sessions


def render_upload_gate(sid: str, sess: dict) -> bool:
    """Renders the upload UI if needed. Returns True once the session is
    ready for the chat view."""
    if sess["pdfs_ready"]:
        return True

    st.subheader("Upload PDF(s) to start this chat")
    files = st.file_uploader(
        "Upload one or more PDFs", type=["pdf"],
        accept_multiple_files=True, key=f"uploader_{sid}",
    )

    if files and st.button("Process PDFs"):
        status = st.status("Processing PDFs…", expanded=True)
        total_chunks = 0

        for f in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name
            try:
                total_chunks += process_pdf(tmp_path, sid, f.name, on_progress=status.write)
                sess["pdf_names"].append(f.name)
            finally:
                os.unlink(tmp_path)

        status.update(
            label=f"✅ Processed {len(files)} PDF(s) — {total_chunks} text chunks indexed.",
            state="complete",
        )
        sess["pdfs_ready"] = True
        save_sessions(st.session_state.sessions)
        st.rerun()

    return False
