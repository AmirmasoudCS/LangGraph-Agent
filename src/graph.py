from pydantic import BaseModel
from langchain_core.messages import (
    AIMessage, HumanMessage, SystemMessage, ToolMessage, trim_messages,
)
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from src.config import (
    get_llm, MAX_REVISIONS, MAX_HISTORY_TOKENS,
    SUMMARY_TRIGGER_TOKENS, KEEP_RECENT_TOKENS,
)
from src.facts import facts_for_prompt
from src.memory import checkpointer
from src.state import AgentState
from src.tools import ALL_TOOLS

llm = get_llm()
llm_with_tools = llm.bind_tools(ALL_TOOLS)

SYSTEM_TEXT = (
    "You are a helpful assistant. Use tools for arithmetic, weather, "
    "the current date/time, and for recent or uncertain facts. Answer concisely."
)
FEEDBACK_PREFIX = "CRITIC FEEDBACK:"


# ---------- Helpers for summarization ----------
def live_messages(state: AgentState):
    """Messages that are NOT yet covered by the summary."""
    msgs = state["messages"]
    upto = state.get("summarized_upto")
    if upto:
        for i, m in enumerate(msgs):
            if m.id == upto:
                return msgs[i + 1:]
    return msgs


def _turn_starts(messages):
    """Indices of real user questions (not critic feedback). Cutting only at
    these points never separates a tool call from its tool result."""
    return [
        i for i, m in enumerate(messages)
        if isinstance(m, HumanMessage) and not m.content.startswith(FEEDBACK_PREFIX)
    ]


def _transcript(messages) -> str:
    lines = []
    for m in messages:
        if isinstance(m, HumanMessage):
            if not m.content.startswith(FEEDBACK_PREFIX):
                lines.append(f"User: {m.content}")
        elif isinstance(m, AIMessage) and m.content:
            lines.append(f"Assistant: {m.content}")
        elif isinstance(m, ToolMessage):
            lines.append(f"Tool ({m.name}): {str(m.content)[:300]}")
    return "\n".join(lines)


# ---------- Node: summarizer (runs before every question) ----------
def summarize(state: AgentState):
    live = live_messages(state)
    if count_tokens_approximately(live) < SUMMARY_TRIGGER_TOKENS:
        return {}                                   # still small: do nothing

    # Choose the cut: keep as many recent turns as fit in KEEP_RECENT_TOKENS,
    # but always keep the current (last) turn.
    starts = _turn_starts(live)
    cut = starts[-1] if starts else 0
    for s in reversed(starts[:-1]):
        if count_tokens_approximately(live[s:]) <= KEEP_RECENT_TOKENS:
            cut = s
        else:
            break
    if cut == 0:
        return {}                                   # nothing old enough to compress

    old = live[:cut]
    previous = state.get("summary", "")
    prompt = (
        "You maintain a running summary of a conversation between a user and an "
        "assistant. Update the summary below with the new messages. Keep every "
        "durable fact: names, preferences, numbers, decisions, results of tool "
        "calls, and unfinished tasks. Drop small talk. Maximum 200 words. "
        "Reply with ONLY the summary.\n\n"
        f"CURRENT SUMMARY:\n{previous or '(empty)'}\n\n"
        f"NEW MESSAGES:\n{_transcript(old)}"
    )
    try:
        new_summary = llm.invoke(prompt).content.strip()
    except Exception:
        return {}                                   # never block the answer
    if not new_summary:
        return {}
    return {"summary": new_summary, "summarized_upto": old[-1].id}


# ---------- Node: the critic ----------
class Verdict(BaseModel):
    ok: bool
    reason: str

critic_llm = llm.with_structured_output(Verdict)

def _last_question_index(messages):
    """Find the user's real question (skipping our own feedback messages)."""
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if isinstance(m, HumanMessage) and not m.content.startswith(FEEDBACK_PREFIX):
            return i
    return 0

def critic(state: AgentState):
    msgs = state["messages"]
    start = _last_question_index(msgs)
    question = msgs[start].content
    evidence = "\n".join(m.content for m in msgs[start:] if isinstance(m, ToolMessage))
    answer = msgs[-1].content

    prompt = (
        "You are a strict reviewer. Decide if the ANSWER correctly and completely "
        "answers the QUESTION. If tool results are given, the answer must agree with them.\n\n"
        f"QUESTION: {question}\n\nTOOL RESULTS: {evidence or '(none)'}\n\nANSWER: {answer}\n\n"
        "Reply with ok=true if the answer is correct and complete, otherwise ok=false "
        "with a short reason explaining what to fix."
    )
    try:
        v = critic_llm.invoke(prompt)
    except Exception:
        return {"verdict": "ok"}          # if the critic breaks, don't block the answer

    if v.ok:
        return {"verdict": "ok"}
    if state["iterations"] >= MAX_REVISIONS:
        return {"verdict": "gave_up"}
    return {
        "verdict": "retry",
        "iterations": state["iterations"] + 1,
        "messages": [HumanMessage(content=f"{FEEDBACK_PREFIX} {v.reason} Please fix your answer.")],
    }

def after_critic(state: AgentState):
    return "agent" if state["verdict"] == "retry" else END


# ---------- Graph factory ----------
_DEFAULT_SAVER = object()   # sentinel: "use the SQLite checkpointer"


def build_graph(use_critic: bool = True, use_facts: bool = True, saver=_DEFAULT_SAVER):
    """Build the agent graph.

    use_critic: False removes the critic node (agent answers go straight to END).
    use_facts:  False keeps long-term facts out of the system prompt.
    saver:      checkpointer to use. Pass MemorySaver() for throw-away runs
                (benchmark), so the real database is never touched.
    """
    if saver is _DEFAULT_SAVER:
        saver = checkpointer

    def agent(state: AgentState):
        system_text = SYSTEM_TEXT
        if state.get("summary"):
            system_text += f"\n\nSummary of the earlier conversation:\n{state['summary']}"

        if use_facts:
            known = facts_for_prompt()
            if known:
                system_text += f"\n\nKnown facts about the user (from earlier sessions):\n{known}"

        # Safety net: trim what the model SEES; the database keeps everything.
        history = trim_messages(
            live_messages(state),
            strategy="last",
            max_tokens=MAX_HISTORY_TOKENS,
            token_counter=count_tokens_approximately,
            start_on="human",
            allow_partial=False,
        )
        reply = llm_with_tools.invoke([SystemMessage(content=system_text)] + history)
        return {"messages": [reply]}

    builder = StateGraph(AgentState)
    builder.add_node("summarize", summarize)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(ALL_TOOLS))

    builder.add_edge(START, "summarize")
    builder.add_edge("summarize", "agent")
    builder.add_edge("tools", "agent")

    if use_critic:
        builder.add_node("critic", critic)
        builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: "critic"})
        builder.add_conditional_edges("critic", after_critic, {"agent": "agent", END: END})
    else:
        builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})

    return builder.compile(checkpointer=saver)


# Default graph: critic on, facts on, SQLite memory
graph = build_graph()


def ask(question: str, thread_id: str = "default", g=None):
    """Run the agent once. Returns (answer, revisions, verdict)."""
    if g is None:
        g = graph
    result = g.invoke(
        {"messages": [("user", question)], "iterations": 0, "verdict": ""},
        config={"configurable": {"thread_id": thread_id}},
    )
    return result["messages"][-1].content, result["iterations"], result["verdict"]