from langchain_ollama import ChatOllama

MODEL_NAME = "gemma4:e4b"
MAX_REVISIONS = 2   # how many times the critic may send the agent back

NUM_CTX = 16384     # context window we ask Ollama for

# Summarization: when the live (unsummarized) history passes the trigger,
# everything older than the most recent KEEP_RECENT_TOKENS is compressed.
SUMMARY_TRIGGER_TOKENS = int(NUM_CTX * 0.4)
KEEP_RECENT_TOKENS = int(NUM_CTX * 0.2)

# Safety net: hard cap on history sent to the model, in case summarizing
# didn't run or a single turn is huge.
MAX_HISTORY_TOKENS = int(NUM_CTX * 0.6)

# Server-side settings, applied only when our program starts the server
OLLAMA_ENV = {
    "OLLAMA_FLASH_ATTENTION": "1",
    "OLLAMA_KV_CACHE_TYPE": "q8_0",
}


def get_llm():
    return ChatOllama(model=MODEL_NAME, temperature=0, num_ctx=NUM_CTX)