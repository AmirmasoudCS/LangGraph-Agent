from langchain_ollama import ChatOllama

MODEL_NAME = "gemma4:e4b"
MAX_REVISIONS = 2

NUM_CTX = 16384
# Leave room for the system prompt, tool schemas, tool results and the reply
MAX_HISTORY_TOKENS = int(NUM_CTX * 0.6)

def get_llm():
    return ChatOllama(model=MODEL_NAME, temperature=0, num_ctx=NUM_CTX)