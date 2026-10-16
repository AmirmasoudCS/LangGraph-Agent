import argparse
import sys

from rich.panel import Panel
from rich.text import Text

from src import facts, sessions
from src.commands import ChatState, dispatch, discard_unsaved, show_sessions
from src.graph import ask
from src.ollama_server import ensure_server
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

    # Feature switches (combine with anything above or below)
    parser.add_argument("--no-critic", "-nc", action="store_true",
                        help="turn the self-fixing critic loop off")
    parser.add_argument("--no-facts", "-nf", action="store_true",
                        help="turn long-term facts off (not read into the prompt, not extracted)")

    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--new", "-n", action="store_true",
                         help="start a new session (named after your first question)")
    actions.add_argument("--session", "-s", metavar="ID_OR_NAME",
                         help="resume a specific session")
    actions.add_argument("--list", "-l", action="store_true",
                         help="list saved sessions and pick one to open")
    actions.add_argument("--remove-session", "-rm", dest="remove", metavar="ID_OR_NAME",
                         help="delete a session and exit")
    actions.add_argument("--memory", "-m", action="store_true",
                         help="show saved facts about you")
    actions.add_argument("--forget", "-f", metavar="ID_OR_ALL",
                         help="delete a saved fact by id, or 'all'")
    return parser.parse_args()


def pick_session():
    """Show the list and let the user open or delete a session.
    Returns the chosen session dict, or None to exit."""
    while True:
        all_sessions = show_sessions()
        if not all_sessions or not sys.stdin.isatty():    # nothing to pick / piped input
            return None

        choice = console.input(
            "\n[bold]Open[/bold] a session by number, [bold]d <number>[/bold] to delete, "
            "or press Enter to quit: "
        ).strip().lower()

        if not choice:
            return None

        deleting = choice.startswith("d ")
        number = choice[2:].strip() if deleting else choice
        if not number.isdigit() or not (1 <= int(number) <= len(all_sessions)):
            console.print(f"[bold red]Enter a number between 1 and {len(all_sessions)}.[/bold red]")
            continue

        selected = all_sessions[int(number) - 1]
        if deleting:
            remove_session(selected["id"])
            console.print()                               # then loop and show the updated list
            continue
        return selected


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

    if args.list:                                         # pick instead of just printing
        try:
            chosen = pick_session()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Bye.[/dim]")
            return
        if chosen is None:
            return
        args.session = chosen["id"]                       # fall through to the normal chat startup
    if args.remove:
        remove_session(args.remove)
        return
    if args.memory:
        rows = facts.all_facts()
        if not rows:
            console.print("[dim]No saved facts yet.[/dim]")
        for r in rows:
            console.print(f"[bold yellow]{r['id']}[/bold yellow]  {r['fact']}")
        return
    if args.forget:
        if args.forget.lower() == "all":
            facts.clear_facts()
        elif args.forget.isdigit():
            facts.delete_fact(int(args.forget))
        else:
            console.print("[bold red]Use a fact id or 'all'.[/bold red]")
            return
        console.print("[green]Done.[/green]")
        return

    try:
        status = ensure_server()
    except RuntimeError as e:
        console.print(f"[bold red]{e}[/bold red]")
        sys.exit(1)
    if status == "started":
        console.print("[dim]Started Ollama with flash attention and q8_0 KV cache.[/dim]")
    else:
        console.print(
            "[yellow]Ollama was already running, so flash attention / KV cache settings "
            "were NOT applied. Quit Ollama from the tray and rerun to apply them.[/yellow]"
        )

    session_id, session_name = resolve_session(args)
    state = ChatState(                                    # everything the /commands can change
        session_id=session_id, session_name=session_name,
        internal=args.internal, use_critic=not args.no_critic, use_facts=not args.no_facts,
    )

    label = f"[bold]{session_name}[/bold] ({session_id})" if session_name else "[bold]new session[/bold]"
    console.print(f"[bold]Agent ready.[/bold] Session: {label}. Type [bold]/help[/bold] for commands, 'quit' to exit.")
    console.print(
        f"[dim]internal messages:[/dim] [bold]{'ON' if state.internal else 'OFF'}[/bold]  "
        f"[dim]critic:[/dim] [bold]{'ON' if state.use_critic else 'OFF'}[/bold]  "
        f"[dim]long-term facts:[/dim] [bold]{'ON' if state.use_facts else 'OFF'}[/bold]"
    )

    try:
        while True:
            q = console.input("\n[bold blue]You:[/bold blue] ")
            text = q.strip()
            if text.lower() in ("quit", "exit"):
                break
            if not text:
                continue
            if text.startswith("/"):                      # slash command, not a question
                if dispatch(text, state) == "quit":
                    break
                continue

            try:
                if state.internal:
                    answer, revisions, verdict = run_with_internals(q, state.session_id, state.graph)
                else:
                    with console.status("[bold cyan]thinking...", spinner="dots"):
                        answer, revisions, verdict = ask(q, state.session_id, state.graph)
            except Exception as e:
                console.print(Panel(
                    Text(f"{type(e).__name__}: {e}", style="white"),
                    title="[bold red]Error[/bold red]",
                    title_align="left",
                    subtitle="[dim]Is Ollama running? You can try the question again.[/dim]",
                    subtitle_align="left",
                    border_style="red",
                    padding=(1, 2),
                ))
                discard_unsaved(state.session_id, state.session_name)
                continue

            style = VERDICT_STYLE.get(verdict, "white")
            border = {"ok": "green", "retry": "red", "gave_up": "yellow"}.get(verdict, "white")
            check = verdict or ("off" if not state.use_critic else "n/a")

            console.print()
            console.print(Panel(
                Text(answer, style="white"),
                title="[bold green]Agent[/bold green]",
                title_align="left",
                subtitle=f"[dim]self-check:[/dim] [{style}]{check}[/{style}] [dim]| revisions: {revisions}[/dim]",
                subtitle_align="left",
                border_style=border,
                padding=(1, 2),
            ))

            # Long-term memory: extract durable facts about the user
            if state.use_facts:
                with console.status("[bold cyan]updating memory...", spinner="dots"):
                    new_facts = facts.extract_and_save(q, answer, state.session_id)
                for f in new_facts:
                    console.print(f"[dim]remembered:[/dim] [italic]{f}[/italic]")

            # First question of a new session: let the agent name it, then save it
            if state.session_name is None:
                with console.status("[bold cyan]naming session...", spinner="dots"):
                    state.session_name = sessions.generate_name(q)
                sessions.add(state.session_id, state.session_name)
                console.print(f"[dim]Session saved as[/dim] [bold]{state.session_name}[/bold] [dim]({state.session_id})[/dim]")
            else:
                sessions.touch(state.session_id)

    except (KeyboardInterrupt, EOFError):
        discard_unsaved(state.session_id, state.session_name)
        console.print("\n[dim]Bye.[/dim]")


if __name__ == "__main__":
    main()