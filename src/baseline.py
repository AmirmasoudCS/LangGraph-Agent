from src.config import get_llm

llm = get_llm()


def ask_baseline(question: str, config: dict | None = None) -> str:
    """A plain LLM call: no tools, no memory, no critic.
    `config` lets the benchmark attach a callback that counts calls and tokens."""
    return llm.invoke(question, config=config).content