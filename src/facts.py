from pydantic import BaseModel

from src.config import get_llm
from src.memory import conn

conn.execute("""
    CREATE TABLE IF NOT EXISTS facts (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        fact       TEXT NOT NULL UNIQUE,
        session_id TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
""")
conn.commit()

MAX_FACTS_IN_PROMPT = 30


def all_facts() -> list[dict]:
    rows = conn.execute("SELECT id, fact FROM facts ORDER BY id").fetchall()
    return [{"id": r[0], "fact": r[1]} for r in rows]


def add_fact(fact: str, session_id: str | None = None) -> bool:
    try:
        cur = conn.execute(
            "INSERT OR IGNORE INTO facts (fact, session_id) VALUES (?, ?)",
            (fact.strip(), session_id),
        )
        conn.commit()
        return cur.rowcount > 0
    except Exception:
        return False


def delete_fact(fact_id: int) -> None:
    conn.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
    conn.commit()


def clear_facts() -> None:
    conn.execute("DELETE FROM facts")
    conn.commit()


def facts_for_prompt() -> str:
    facts = all_facts()[-MAX_FACTS_IN_PROMPT:]
    return "\n".join(f"- {f['fact']}" for f in facts)


# ---------- Extraction ----------
class NewFacts(BaseModel):
    facts: list[str]


_extractor = get_llm().with_structured_output(NewFacts)


def extract_and_save(question: str, answer: str, session_id: str) -> list[str]:
    """Look at one exchange and store durable facts about the user."""
    known = facts_for_prompt() or "(none)"
    prompt = (
        "Extract durable facts about the USER from this exchange: their name, "
        "location, job, preferences, ongoing projects. Rules:\n"
        "- Only facts the USER stated about themselves. Never facts from the assistant's answer.\n"
        "- Skip questions, one-off requests, and temporary details (weather, time).\n"
        "- Each fact is one short standalone sentence, e.g. 'The user's name is Sam.'\n"
        "- Do not repeat facts already known.\n"
        "- If there is nothing worth saving, return an empty list.\n\n"
        f"ALREADY KNOWN:\n{known}\n\n"
        f"USER MESSAGE: {question}\nASSISTANT ANSWER: {answer}"
    )
    try:
        result = _extractor.invoke(prompt)
    except Exception:
        return []                                  # never break the chat
    saved = []
    for f in result.facts:
        f = f.strip()
        if f and len(f) <= 200 and add_fact(f, session_id):
            saved.append(f)
    return saved