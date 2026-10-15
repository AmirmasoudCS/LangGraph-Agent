import uuid
from datetime import datetime

from src.config import get_llm
from src.memory import conn, checkpointer

conn.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        id         TEXT PRIMARY KEY,
        name       TEXT NOT NULL,
        created_at TEXT NOT NULL,
        last_used  TEXT NOT NULL
    )
""")
conn.commit()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def new_id() -> str:
    return uuid.uuid4().hex[:8]


def add(session_id: str, name: str) -> None:
    conn.execute(
        "INSERT INTO sessions (id, name, created_at, last_used) VALUES (?, ?, ?, ?)",
        (session_id, name, _now(), _now()),
    )
    conn.commit()


def touch(session_id: str) -> None:
    conn.execute("UPDATE sessions SET last_used = ? WHERE id = ?", (_now(), session_id))
    conn.commit()


def list_all() -> list[dict]:
    rows = conn.execute(
        "SELECT id, name, created_at, last_used FROM sessions ORDER BY last_used DESC"
    ).fetchall()
    return [dict(zip(("id", "name", "created_at", "last_used"), r)) for r in rows]


def latest() -> dict | None:
    all_sessions = list_all()
    return all_sessions[0] if all_sessions else None


def find(ref: str) -> dict | None:
    """Find a session by exact id, id prefix, or name (case-insensitive).
    Raises ValueError if the reference matches more than one session."""
    ref = ref.strip().lower()
    sessions = list_all()

    for s in sessions:                      # exact id wins
        if s["id"] == ref:
            return s
    matches = [s for s in sessions if s["id"].startswith(ref) or s["name"].lower() == ref]
    if len(matches) > 1:
        raise ValueError(f"'{ref}' matches several sessions; use the id instead.")
    return matches[0] if matches else None


def remove(session_id: str) -> None:
    conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    conn.commit()
    checkpointer.delete_thread(session_id)  # erase the saved conversation too


def generate_name(question: str) -> str:
    """Ask the LLM for a short title based on the first question."""
    fallback = question.strip()[:40] or "Untitled"
    prompt = (
        "Write a short title (maximum 5 words) for a chat that starts with the "
        "message below. Reply with ONLY the title: no quotes, no period.\n\n"
        f"Message: {question}"
    )
    try:
        title = get_llm().invoke(prompt).content.strip().splitlines()[0]
        return title.strip("\"'*# ")[:50] or fallback
    except Exception:
        return fallback