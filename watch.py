from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage

from src.graph import graph

console = Console()

NODE_STYLE = {"agent": "cyan", "tools": "yellow", "critic": "magenta"}
NODE_ICON = {"agent": "🧠", "tools": "🔧", "critic": "🔍"}
VERDICT_STYLE = {"ok": "bold green", "retry": "bold red", "gave_up": "bold yellow"}


def shorten(text: str, limit: int = 600) -> str:
    return text if len(text) <= limit else text[:limit] + " …"


def render_message(m) -> list:
    """Turn one message into a list of colored rich objects."""
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


def watch(question: str, thread_id: str = "watch-1"):
    inputs = {"messages": [("user", question)], "iterations": 0, "verdict": ""}
    config = {"configurable": {"thread_id": thread_id}}
    step_no = 0

    with console.status("[bold cyan]thinking...", spinner="dots"):
        for step in graph.stream(inputs, config=config, stream_mode="updates"):
            for node, update in step.items():
                step_no += 1
                body = []
                for m in update.get("messages", []):
                    body += render_message(m)

                if "verdict" in update:
                    style = VERDICT_STYLE.get(update["verdict"], "white")
                    body.append(Text(f"verdict: {update['verdict']}", style=style))
                if "iterations" in update:
                    body.append(Text(f"revisions so far: {update['iterations']}", style="red"))

                color = NODE_STYLE.get(node, "white")
                console.print(Panel(
                    Group(*body) if body else Text("(no output)", style="dim"),
                    title=f"{NODE_ICON.get(node, '•')} step {step_no}: {node.upper()}",
                    title_align="left",
                    border_style=color,
                ))


if __name__ == "__main__":
    console.print("[bold]Agent watcher[/bold] - type 'quit' to exit\n")
    while True:
        q = console.input("[bold blue]You:[/bold blue] ")
        if q.strip().lower() in ("quit", "exit"):
            break
        watch(q)