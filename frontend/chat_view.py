"""Chat interface: renders message history and handles new turns."""

import streamlit as st

from backend.chain import ask
from backend.session_store import save_sessions


def render_chat(sid: str, sess: dict) -> None:
    st.caption("📎 " + ", ".join(sess["pdf_names"]))

    for msg in sess["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("Ask something about your document(s)…")
    if not query:
        return

    sess["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            answer = ask(sid, query)
        st.markdown(answer)

    sess["messages"].append({"role": "assistant", "content": answer})
    save_sessions(st.session_state.sessions)
