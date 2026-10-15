"""One interface for every benchmark condition.

Each runner takes a question and returns a RunResult, so the benchmark loop
can iterate over CONDITIONS without special cases.

Conditions:
  baseline         plain LLM, no tools, no memory, no critic
  agent_no_critic  agent with tools, summarizer and loop, but no critic
  agent_full       agent with tools and the self-fixing critic loop

Long-term facts are OFF in all agent conditions, so your personal facts never
leak into benchmark runs. Every question gets a fresh thread and an in-memory
checkpointer, so the real database is never touched.
"""
import time
from dataclasses import dataclass, field, asdict
from uuid import uuid4

from langchain_core.callbacks import BaseCallbackHandler
from langgraph.checkpoint.memory import MemorySaver

from src.baseline import ask_baseline
from src.graph import build_graph


class Counter(BaseCallbackHandler):
    """Counts model calls, tool calls and (best effort) tokens during one run,
    and records which tools ran and what they returned."""

    def __init__(self):
        self.llm_calls = 0
        self.tool_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.tools_used: list[str] = []
        self.tool_outputs: list[str] = []

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self.llm_calls += 1

    def on_llm_start(self, serialized, prompts, **kwargs):
        pass                                   # chat models report via on_chat_model_start

    def on_tool_start(self, serialized, input_str, **kwargs):
        self.tool_calls += 1
        name = (serialized or {}).get("name") or kwargs.get("name") or "unknown"
        self.tools_used.append(name)

    def on_tool_end(self, output, **kwargs):
        content = getattr(output, "content", output)
        self.tool_outputs.append(str(content)[:1500])

    def on_llm_end(self, response, **kwargs):
        for generations in response.generations:
            for g in generations:
                usage = getattr(getattr(g, "message", None), "usage_metadata", None)
                if usage:
                    self.input_tokens += usage.get("input_tokens", 0)
                    self.output_tokens += usage.get("output_tokens", 0)


@dataclass
class RunResult:
    condition: str
    answer: str = ""
    latency_s: float = 0.0
    llm_calls: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    revisions: int = 0
    verdict: str = ""
    error: str = ""
    tools_used: list = field(default_factory=list)
    tool_outputs: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------- Runners ----------
def run_baseline(question: str) -> RunResult:
    counter = Counter()
    result = RunResult(condition="baseline")
    start = time.perf_counter()
    try:
        result.answer = ask_baseline(question, config={"callbacks": [counter]})
    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
    result.latency_s = round(time.perf_counter() - start, 2)
    _fill(result, counter)
    return result


_graphs: dict = {}


def _get_graph(use_critic: bool):
    """Build each benchmark graph once and reuse it (facts off, in-memory saver)."""
    if use_critic not in _graphs:
        _graphs[use_critic] = build_graph(
            use_critic=use_critic, use_facts=False, saver=MemorySaver()
        )
    return _graphs[use_critic]


def _run_agent(question: str, use_critic: bool, name: str) -> RunResult:
    counter = Counter()
    result = RunResult(condition=name)
    config = {
        "configurable": {"thread_id": uuid4().hex},   # fresh thread: no shared history
        "callbacks": [counter],
        "recursion_limit": 25,
    }
    start = time.perf_counter()
    try:
        out = _get_graph(use_critic).invoke(
            {"messages": [("user", question)], "iterations": 0, "verdict": ""},
            config=config,
        )
        result.answer = out["messages"][-1].content
        result.revisions = out.get("iterations", 0)
        result.verdict = out.get("verdict", "") or "n/a"
    except Exception as e:                            # e.g. GraphRecursionError
        result.error = f"{type(e).__name__}: {e}"
    result.latency_s = round(time.perf_counter() - start, 2)
    _fill(result, counter)
    return result


def run_agent_no_critic(question: str) -> RunResult:
    return _run_agent(question, use_critic=False, name="agent_no_critic")


def run_agent_full(question: str) -> RunResult:
    return _run_agent(question, use_critic=True, name="agent_full")


def _fill(result: RunResult, counter: Counter) -> None:
    result.llm_calls = counter.llm_calls
    result.tool_calls = counter.tool_calls
    result.input_tokens = counter.input_tokens
    result.output_tokens = counter.output_tokens
    result.tools_used = counter.tools_used
    result.tool_outputs = counter.tool_outputs


CONDITIONS = {
    "baseline": run_baseline,
    "agent_no_critic": run_agent_no_critic,
    "agent_full": run_agent_full,
}