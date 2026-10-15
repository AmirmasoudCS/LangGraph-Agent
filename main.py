import argparse

from src.graph import ask
from src.display import console, run_with_internals, VERDICT_STYLE


def parse_args():
    parser = argparse.ArgumentParser(description="LangGraph agent with tools, memory and a self-fixing loop.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--internal-messages-on", "-imon",
        dest="internal", action="store_const", const=True,
        help="show every node step (agent, tools, critic) as it runs",
    )
    group.add_argument(
        "--internal-messages-off", "-imoff",
        dest="internal", action="store_const", const=False,
        help="show only the final answer (default)",
    )
    parser.set_defaults(internal=False)
    return parser.parse_args()


def main():
    args = parse_args()
    mode = "ON" if args.internal else "OFF"
    console.print(f"[bold]Agent ready.[/bold] Internal messages: [bold]{mode}[/bold]. Type 'quit' to exit.")

    thread_id = "session-1"
    while True:
        q = console.input("\n[bold blue]You:[/bold blue] ")
        if q.strip().lower() in ("quit", "exit"):
            break
        if not q.strip():
            continue

        if args.internal:
            answer, revisions, verdict = run_with_internals(q, thread_id)
        else:
            with console.status("[bold cyan]thinking...", spinner="dots"):
                answer, revisions, verdict = ask(q, thread_id)

        console.print(f"\n[bold green]Agent:[/bold green] {answer}")
        style = VERDICT_STYLE.get(verdict, "white")
        console.print(f"[dim]self-check:[/dim] [{style}]{verdict or 'n/a'}[/{style}] [dim]| revisions: {revisions}[/dim]")


if __name__ == "__main__":
    main()