"""Session + chat-message persistence. Plain JSON file, single-process only —
see README for swapping this for SQLite in a multi-user deployment."""

import json
import os
import uuid

from backend.config import SESSIONS_FILE


def load_sessions() -> dict:
    if os.path.exists(SESSIONS_FILE):
        try:
            with open(SESSIONS_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_sessions(sessions: dict) -> None:
    os.makedirs(os.path.dirname(SESSIONS_FILE) or ".", exist_ok=True)
    with open(SESSIONS_FILE, "w") as f:
        json.dump(sessions, f, indent=2)


def create_session(sessions: dict) -> str:
    """Mutates `sessions` in place and returns the new session id."""
    sid = uuid.uuid4().hex[:8]
    sessions[sid] = {
        "name": f"Chat {len(sessions) + 1}",
        "messages": [],
        "pdfs_ready": False,
        "pdf_names": [],
    }
    return sid


def delete_session(sessions: dict, sid: str) -> None:
    sessions.pop(sid, None)
