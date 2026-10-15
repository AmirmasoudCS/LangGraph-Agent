from src.graph import graph

def watch(question: str, thread_id: str = "watch-1"):
    inputs = {"messages": [("user", question)], "iterations": 0, "verdict": ""}
    config = {"configurable": {"thread_id": thread_id}}

    for step in graph.stream(inputs, config=config, stream_mode="updates"):
        for node_name, update in step.items():
            print(f"\n=== NODE: {node_name} ===")
            for m in update.get("messages", []):
                m.pretty_print()
            for key in ("iterations", "verdict"):
                if key in update:
                    print(f"[state update] {key} = {update[key]!r}")

if __name__ == "__main__":
    while True:
        q = input("\nYou: ")
        if q.strip().lower() in ("quit", "exit"):
            break
        watch(q)