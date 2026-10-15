from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage

from src.graph import graph

console = Console()

NODE_STYLE = {"summarize": "blue", "agent": "cyan", "tools": "yellow", "critic": "magenta"}
VERDICT_STYLE = {"ok": "bold green", "retry": "bold red", "gave_up": "bold yellow"}


def shorten(text: str, limit: int = 600) -> str:
    return text if len(text) <= limit else text[:limit] + " …"


def render_message(m) -> list:
    parts = []
    if isinstance(m, AIMessage):
        for tc in m.tool_calls:
            args = ", ".join(f"{k}={v!r}" for k, v in tc["args"].items())
            parts.append(Text(f"calls {tc['name']}({args})", style="bold yellow"))
        if m.content:
            parts.append(Text(m.content, style="white"))
    elif isinstance(m, ToolMessage):
        parts.append(Text(f"{m.name} returned:", style="dim"))
        parts.append(Text(shorten(str(m.content)), style="green"))
    elif isinstance(m, HumanMessage):
        parts.append(Text(m.content, style="bold red"))   # critic feedback
    return parts


def run_with_internals(question: str, thread_id: str):
    """Stream the graph and print one panel per node execution."""
    inputs = {"messages": [("user", question)], "iterations": 0, "verdict": ""}
    config = {"configurable": {"thread_id": thread_id}}
    step_no = 0

    with console.status("[bold cyan]thinking...", spinner="dots"):
        for step in graph.stream(inputs, config=config, stream_mode="updates"):
            for node, update in step.items():
                update = update or {}
                if node == "summarize" and not update:
                    continue                      # summarizer had nothing to do

                step_no += 1
                body = []
                for m in update.get("messages", []):
                    body += render_message(m)
                if "summary" in update:
                    body.append(Text("new running summary:", style="dim"))
                    body.append(Text(update["summary"], style="white"))
                if "verdict" in update:
                    style = VERDICT_STYLE.get(update["verdict"], "white")
                    body.append(Text(f"verdict: {update['verdict']}", style=style))
                if "iterations" in update:
                    body.append(Text(f"revisions so far: {update['iterations']}", style="red"))

                console.print(Panel(
                    Group(*body) if body else Text("(no output)", style="dim"),
                    title=f"step {step_no}: {node.upper()}",
                    title_align="left",
                    border_style=NODE_STYLE.get(node, "white"),
                ))

    # Streaming doesn't return the final result, so read it from the checkpoint
    state = graph.get_state(config).values
    return state["messages"][-1].content, state["iterations"], state["verdict"]