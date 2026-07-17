"""Sidebar: list of chat sessions, new-chat button, delete-chat button."""

import streamlit as st

from backend.session_store import create_session, delete_session, save_sessions


def render_sidebar() -> None:
    with st.sidebar:
        st.title("💬 Chats")

        if st.button("➕ New chat", use_container_width=True):
            sid = create_session(st.session_state.sessions)
            st.session_state.current_session = sid
            save_sessions(st.session_state.sessions)
            st.rerun()

        st.divider()

        for sid, sess in st.session_state.sessions.items():
            label = sess["name"] + (" ✅" if sess["pdfs_ready"] else " ⏳")
            col_select, col_delete = st.columns([4, 1])

            with col_select:
                if st.button(label, key=f"sel_{sid}", use_container_width=True):
                    st.session_state.current_session = sid
                    st.rerun()

            with col_delete:
                if st.button("🗑️", key=f"del_{sid}"):
                    delete_session(st.session_state.sessions, sid)
                    if st.session_state.current_session == sid:
                        remaining = list(st.session_state.sessions)
                        st.session_state.current_session = remaining[-1] if remaining else None
                    save_sessions(st.session_state.sessions)
                    st.rerun()
