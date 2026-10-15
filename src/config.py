from langchain_ollama import ChatOllama

MODEL_NAME = "gemma4:e4b"
MAX_REVISIONS = 2   # how many times the critic may send the agent back

def get_llm():
    return ChatOllama(model=MODEL_NAME, temperature=0, num_ctx=8192)