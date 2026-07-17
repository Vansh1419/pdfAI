"""Entrypoint: `streamlit run app.py`

This file only wires session state to the frontend views. All logic lives
in backend/ (no Streamlit imports there) and frontend/ (Streamlit views,
no business logic there).
"""

import streamlit as st

from backend.session_store import load_sessions
from frontend.chat_view import render_chat
from frontend.sidebar import render_sidebar
from frontend.upload_view import render_upload_gate

st.set_page_config(page_title="Document Chat", layout="wide")

# ── App state (loaded once per browser session, backed by disk) ──────────
if "sessions" not in st.session_state:
    st.session_state.sessions = load_sessions()
if "current_session" not in st.session_state:
    st.session_state.current_session = (
        next(reversed(st.session_state.sessions), None)
        if st.session_state.sessions else None
    )

render_sidebar()

if st.session_state.current_session is None:
    st.info("Create a new chat from the sidebar to get started.")
    st.stop()

sid = st.session_state.current_session
sess = st.session_state.sessions[sid]

st.title(sess["name"])

if render_upload_gate(sid, sess):
    render_chat(sid, sess)
