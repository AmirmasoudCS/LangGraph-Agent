from src.config import get_llm

llm = get_llm()

def ask_baseline(question: str) -> str:
    return llm.invoke(question).content