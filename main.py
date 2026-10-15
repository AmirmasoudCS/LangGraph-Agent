from src.graph import ask

print("Agent ready. Type 'quit' to exit.")
while True:
    q = input("\nYou: ")
    if q.strip().lower() in ("quit", "exit"):
        break
    answer, revisions, verdict = ask(q, thread_id="session-1")
    print(f"\nAgent: {answer}")
    print(f"[self-check: {verdict}, revisions: {revisions}]")