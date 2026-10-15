from pydantic import BaseModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from src.config import get_llm, MAX_REVISIONS, MAX_HISTORY_TOKENS
from src.memory import checkpointer
from src.state import AgentState
from src.tools import ALL_TOOLS

llm = get_llm()
llm_with_tools = llm.bind_tools(ALL_TOOLS)

SYSTEM = SystemMessage(content=(
    "You are a helpful assistant. Use tools for arithmetic, weather, "
    "the current date/time, and for recent or uncertain facts. Answer concisely."
))
FEEDBACK_PREFIX = "CRITIC FEEDBACK:"


# ---------- Node 1: the agent (LLM that can call tools) ----------
def agent(state: AgentState):
    # Trim only what the model SEES; the database keeps the full history.
    history = trim_messages(
        state["messages"],
        strategy="last",                          # keep the most recent messages
        max_tokens=MAX_HISTORY_TOKENS,
        token_counter=count_tokens_approximately,
        start_on="human",                         # never start mid tool-call
        allow_partial=False,                      # never cut a message in half
    )
    reply = llm_with_tools.invoke([SYSTEM] + history)
    return {"messages": [reply]}


# ---------- Node 2: the critic ----------
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


# ---------- Wiring ----------
builder = StateGraph(AgentState)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(ALL_TOOLS))
builder.add_node("critic", critic)

builder.add_edge(START, "agent")
# If the agent asked for tools -> "tools"; if it just answered -> "critic"
builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: "critic"})
builder.add_edge("tools", "agent")
builder.add_conditional_edges("critic", after_critic, {"agent": "agent", END: END})

graph = builder.compile(checkpointer=checkpointer)   # SQLite = memory that survives restarts


def ask(question: str, thread_id: str = "default"):
    """Run the agent once. Returns (answer, revisions, verdict)."""
    result = graph.invoke(
        {"messages": [("user", question)], "iterations": 0, "verdict": ""},
        config={"configurable": {"thread_id": thread_id}},
    )
    return result["messages"][-1].content, result["iterations"], result["verdict"]