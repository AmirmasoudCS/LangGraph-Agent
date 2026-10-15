from pathlib import Path
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

DB_PATH = Path("data/agent.db")
DB_PATH.parent.mkdir(exist_ok=True)

# One connection shared by the checkpointer (conversation memory)
# and the sessions table (names, timestamps).
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(conn)