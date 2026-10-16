"""In-chat slash commands: /help, /imon, /imoff, /critic, /facts, /doc, ...

How it works
  * @command registers a function under a name (and optional aliases).
  * dispatch() is called by main.py for every input line that starts with "/".
  * Commands receive the shared ChatState, so they can change the running chat
    (internal messages on/off, critic on/off, switch session, ...).
  * A command returns "quit" to leave the chat, or None to carry on.

To add a command, write a function and decorate it. Nothing else needs to change.
"""
import difflib
from dataclasses import dataclass

from rich.markup import escape
from rich.table import Table

from src import facts, sessions
from src.display import console
from src.graph import build_graph
from src.memory import checkpointer

try:                                     # optional feature, built later
    from src import documents
except ImportError:
    documents = None


# =====================================================================
# Shared chat state
# =====================================================================
@dataclass
class ChatState:
    session_id: str
    session_name: str | None             # None = brand-new session, not saved yet
    internal: bool = False
    use_critic: bool = True
    use_facts: bool = True
    graph: object = None

    def __post_init__(self):
        self.rebuild()

    def rebuild(self):
        """Rebuild the graph after the critic/facts switches change.
        Memory lives in the SQLite checkpointer, so the conversation continues."""
        self.graph = build_graph(use_critic=self.use_critic, use_facts=self.use_facts)


# =====================================================================
# Helpers shared with main.py
# =====================================================================
def discard_unsaved(session_id: str, session_name):
    """A brand-new session that never got registered must not leave
    orphan checkpoints behind in the database."""
    if session_name is None:
        checkpointer.delete_thread(session_id)


def show_sessions() -> list[dict]:
    all_sessions = sessions.list_all()
    if not all_sessions:
        console.print("[dim]No saved sessions yet.[/dim]")
        return []
    table = Table(title="Sessions", title_justify="left", border_style="cyan")
    table.add_column("#", style="bold green", justify="right")
    table.add_column("ID", style="bold yellow")
    table.add_column("Name")
    table.add_column("Created", style="dim")
    table.add_column("Last used", style="dim")
    for i, s in enumerate(all_sessions, start=1):
        table.add_row(str(i), s["id"], s["name"],
                      s["created_at"].replace("T", " "), s["last_used"].replace("T", " "))
    console.print(table)
    return all_sessions


# =====================================================================
# Registry
# =====================================================================
COMMANDS: dict[str, dict] = {}


def command(name: str, *aliases: str, usage: str | None = None, help: str = ""):
    def register(fn):
        info = {"name": name, "fn": fn, "usage": usage or f"/{name}", "help": help}
        for n in (name, *aliases):
            COMMANDS[n] = info
        return fn
    return register


def dispatch(line: str, state: ChatState):
    """Run one slash command. Returns "quit" or None."""
    name, _, arg = line[1:].strip().partition(" ")
    name = name.lower()
    if name == "":
        name = "help"
    info = COMMANDS.get(name)
    if info is None:
        close = difflib.get_close_matches(name, COMMANDS, n=1)
        hint = f" Did you mean [bold]/{close[0]}[/bold]?" if close else ""
        console.print(f"[red]Unknown command /{name}.[/red]{hint} Type [bold]/help[/bold].")
        return None
    return info["fn"](state, arg.strip())


def _toggle(arg: str, current: bool):
    """'on' -> True, 'off' -> False, '' -> flip, anything else -> None (invalid)."""
    a = arg.lower()
    if a in ("on", "1", "true"):
        return True
    if a in ("off", "0", "false"):
        return False
    return (not current) if a == "" else None


def _onoff(value: bool) -> str:
    return "[bold green]ON[/bold green]" if value else "[bold red]OFF[/bold red]"


# =====================================================================
# Commands
# =====================================================================
@command("help", "h", "?", help="list all commands")
def cmd_help(state, arg):
    table = Table(title="Commands", title_justify="left", border_style="cyan")
    table.add_column("Command", style="bold yellow")
    table.add_column("What it does")
    seen = set()
    for info in COMMANDS.values():
        if info["name"] not in seen:
            seen.add(info["name"])
            table.add_row(escape(info["usage"]), escape(info["help"]))
    console.print(table)


@command("status", help="show the current session and switches")
def cmd_status(state, arg):
    label = f"{state.session_name} ({state.session_id})" if state.session_name else f"new session ({state.session_id})"
    console.print(f"[dim]session:[/dim] {label}")
    console.print(f"[dim]internal messages:[/dim] {_onoff(state.internal)}   "
                  f"[dim]critic:[/dim] {_onoff(state.use_critic)}   "
                  f"[dim]long-term facts:[/dim] {_onoff(state.use_facts)}")


@command("imon", help="show internal steps (agent, tools, critic)")
def cmd_imon(state, arg):
    state.internal = True
    console.print(f"Internal messages {_onoff(True)}")


@command("imoff", help="show only the final answer")
def cmd_imoff(state, arg):
    state.internal = False
    console.print(f"Internal messages {_onoff(False)}")


@command("critic", usage="/critic [on|off]", help="turn the self-fixing critic loop on or off")
def cmd_critic(state, arg):
    value = _toggle(arg, state.use_critic)
    if value is None:
        console.print("[red]Usage: /critic on|off[/red]")
        return
    state.use_critic = value
    state.rebuild()
    console.print(f"Critic {_onoff(value)}")


@command("facts", usage="/facts [on|off]", help="turn long-term facts on or off (read and saved)")
def cmd_facts(state, arg):
    value = _toggle(arg, state.use_facts)
    if value is None:
        console.print("[red]Usage: /facts on|off[/red]")
        return
    state.use_facts = value
    state.rebuild()
    console.print(f"Long-term facts {_onoff(value)}")


@command("memory", "mem", help="show saved facts about you")
def cmd_memory(state, arg):
    rows = facts.all_facts()
    if not rows:
        console.print("[dim]No saved facts yet.[/dim]")
    for r in rows:
        console.print(f"[bold yellow]{r['id']}[/bold yellow]  {r['fact']}")


@command("forget", usage="/forget <id|all>", help="delete a saved fact by id, or all of them")
def cmd_forget(state, arg):
    if arg.lower() == "all":
        facts.clear_facts()
    elif arg.isdigit():
        facts.delete_fact(int(arg))
    else:
        console.print("[red]Usage: /forget <id> or /forget all[/red]")
        return
    console.print("[green]Done.[/green]")


@command("sessions", "ls", help="list saved sessions")
def cmd_sessions(state, arg):
    show_sessions()


def _switch(state: ChatState, session_id: str, name: str | None):
    discard_unsaved(state.session_id, state.session_name)   # drop an unused new session
    state.session_id, state.session_name = session_id, name


@command("new", help="start a new session (named after your first question)")
def cmd_new(state, arg):
    _switch(state, sessions.new_id(), None)
    console.print("[green]New session.[/green] It will be named after your first question.")


@command("open", usage="/open <number|id|name>", help="switch to another saved session")
def cmd_open(state, arg):
    if not arg:
        show_sessions()
        console.print("[dim]Use /open <number|id|name>[/dim]")
        return
    all_sessions = sessions.list_all()
    target = None
    if arg.isdigit() and 1 <= int(arg) <= len(all_sessions):   # number from /sessions
        target = all_sessions[int(arg) - 1]
    else:
        try:
            target = sessions.find(arg)
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            return
    if target is None:
        console.print(f"[red]No session matches '{arg}'.[/red]")
        return
    if target["id"] == state.session_id:
        console.print("[dim]Already in that session.[/dim]")
        return
    _switch(state, target["id"], target["name"])
    console.print(f"[green]Switched to[/green] [bold]{target['name']}[/bold] ({target['id']})")


@command("doc", "docs", usage="/doc add <path> | list | remove <id>",
         help="add, list or remove documents for the agent to search")
def cmd_doc(state, arg):
    if documents is None:
        console.print("[yellow]Document search is not installed yet "
                      "(src/documents.py does not exist).[/yellow]")
        return
    sub, _, rest = arg.partition(" ")
    sub = sub.lower()
    rest = rest.strip().strip('"').strip("'")           # drag-and-drop pastes a quoted path
    if sub == "add" and rest:
        with console.status("[bold cyan]indexing document...", spinner="dots"):
            try:
                info = documents.add_file(rest)
            except Exception as e:
                console.print(f"[red]Could not add the file: {e}[/red]")
                return
        console.print(f"[green]Added[/green] {info['name']} [dim]({info['chunks']} chunks)[/dim]")
    elif sub in ("", "list"):
        docs = documents.list_all()
        if not docs:
            console.print("[dim]No documents yet. Use /doc add <path>.[/dim]")
        for d in docs:
            console.print(f"[bold yellow]{d['id']}[/bold yellow]  {d['name']} [dim]({d['chunks']} chunks)[/dim]")
    elif sub in ("remove", "rm") and rest:
        ok = documents.remove(rest)
        console.print("[green]Removed.[/green]" if ok else f"[red]No document matches '{rest}'.[/red]")
    else:
        console.print("[red]Usage: /doc add <path> | /doc list | /doc remove <id>[/red]")


@command("quit", "exit", "q", help="leave the chat")
def cmd_quit(state, arg):
    return "quit"