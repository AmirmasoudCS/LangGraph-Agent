import argparse
import sys

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src import sessions
from src.graph import ask
from src.display import console, run_with_internals, VERDICT_STYLE


def parse_args():
    parser = argparse.ArgumentParser(description="LangGraph agent with tools, memory and a self-fixing loop.")

    internals = parser.add_mutually_exclusive_group()
    internals.add_argument(
        "--internal-messages-on", "-imon",
        dest="internal", action="store_const", const=True,
        help="show every node step (agent, tools, critic) as it runs",
    )
    internals.add_argument(
        "--internal-messages-off", "-imoff",
        dest="internal", action="store_const", const=False,
        help="show only the final answer (default)",
    )
    parser.set_defaults(internal=False)

    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--new", "-n", action="store_true",
                         help="start a new session (named after your first question)")
    actions.add_argument("--session", "-s", metavar="ID_OR_NAME",
                         help="resume a specific session")
    actions.add_argument("--list", "-l", action="store_true",
                         help="list saved sessions and exit")
    actions.add_argument("--remove-session", "-rm", dest="remove", metavar="ID_OR_NAME",
                         help="delete a session and exit")
    return parser.parse_args()


def show_sessions():
    all_sessions = sessions.list_all()
    if not all_sessions:
        console.print("[dim]No saved sessions yet.[/dim]")
        return
    table = Table(title="Sessions", title_justify="left", border_style="cyan")
    table.add_column("ID", style="bold yellow")
    table.add_column("Name")
    table.add_column("Created", style="dim")
    table.add_column("Last used", style="dim")
    for s in all_sessions:
        table.add_row(s["id"], s["name"], s["created_at"].replace("T", " "), s["last_used"].replace("T", " "))
    console.print(table)


def remove_session(ref: str):
    try:
        s = sessions.find(ref)
    except ValueError as e:
        console.print(f"[bold red]{e}[/bold red]")
        sys.exit(1)
    if not s:
        console.print(f"[bold red]No session matches '{ref}'.[/bold red]")
        sys.exit(1)
    confirm = console.input(f"Delete [bold]{s['name']}[/bold] ({s['id']})? [y/N] ")
    if confirm.strip().lower() == "y":
        sessions.remove(s["id"])
        console.print("[green]Session removed.[/green]")
    else:
        console.print("[dim]Cancelled.[/dim]")


def resolve_session(args):
    """Return (session_id, session_name). name is None for a brand-new session."""
    if args.new:
        return sessions.new_id(), None
    if args.session:
        try:
            s = sessions.find(args.session)
        except ValueError as e:
            console.print(f"[bold red]{e}[/bold red]")
            sys.exit(1)
        if not s:
            console.print(f"[bold red]No session matches '{args.session}'.[/bold red]")
            sys.exit(1)
        return s["id"], s["name"]
    s = sessions.latest()                      # default: resume the most recent one
    if s:
        return s["id"], s["name"]
    return sessions.new_id(), None             # (change this to always start fresh if you prefer)


def main():
    args = parse_args()

    if args.list:
        show_sessions()
        return
    if args.remove:
        remove_session(args.remove)
        return

    session_id, session_name = resolve_session(args)
    mode = "ON" if args.internal else "OFF"
    label = f"[bold]{session_name}[/bold] ({session_id})" if session_name else "[bold]new session[/bold]"
    console.print(f"[bold]Agent ready.[/bold] Session: {label}. Internal messages: [bold]{mode}[/bold]. Type 'quit' to exit.")

    while True:
        q = console.input("\n[bold blue]You:[/bold blue] ")
        if q.strip().lower() in ("quit", "exit"):
            break
        if not q.strip():
            continue

        if args.internal:
            answer, revisions, verdict = run_with_internals(q, session_id)
        else:
            with console.status("[bold cyan]thinking...", spinner="dots"):
                answer, revisions, verdict = ask(q, session_id)

        style = VERDICT_STYLE.get(verdict, "white")
        border = {"ok": "green", "retry": "red", "gave_up": "yellow"}.get(verdict, "white")

        console.print()
        console.print(Panel(
            Text(answer, style="white"),
            title="[bold green]Agent[/bold green]",
            title_align="left",
            subtitle=f"[dim]self-check:[/dim] [{style}]{verdict or 'n/a'}[/{style}] [dim]| revisions: {revisions}[/dim]",
            subtitle_align="left",
            border_style=border,
            padding=(1, 2),
        ))

        # First question of a new session: let the agent name it, then save it
        if session_name is None:
            with console.status("[bold cyan]naming session...", spinner="dots"):
                session_name = sessions.generate_name(q)
            sessions.add(session_id, session_name)
            console.print(f"[dim]Session saved as[/dim] [bold]{session_name}[/bold] [dim]({session_id})[/dim]")
        else:
            sessions.touch(session_id)


if __name__ == "__main__":
    main()