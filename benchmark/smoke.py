"""Quick check that all three benchmark conditions run end to end.

Usage (from the project root):
    python -m benchmark.smoke
"""
from rich.console import Console
from rich.table import Table

from src.ollama_server import ensure_server
from benchmark.runners import CONDITIONS

console = Console()

QUESTIONS = [
    "What is 4821 * 337 + 19?",         # the agent should call the calculator
    "What is the capital of France?",   # no tool needed
]


def main():
    ensure_server()

    table = Table(title="Smoke test", title_justify="left", border_style="cyan")
    table.add_column("Question", max_width=28)
    table.add_column("Condition", style="bold yellow")
    table.add_column("Answer", max_width=40)
    table.add_column("Time (s)", justify="right")
    table.add_column("LLM calls", justify="right")
    table.add_column("Tools", justify="right")
    table.add_column("Revisions", justify="right")
    table.add_column("Error", style="red", max_width=30)

    for q in QUESTIONS:
        for name, run in CONDITIONS.items():
            console.print(f"[dim]running {name}: {q}[/dim]")
            r = run(q)
            table.add_row(
                q, name, r.answer.replace("\n", " ")[:120], str(r.latency_s),
                str(r.llm_calls), str(r.tool_calls), str(r.revisions), r.error,
            )

    console.print(table)


if __name__ == "__main__":
    main()