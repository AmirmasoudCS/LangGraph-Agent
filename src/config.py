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

# ---------- Document search (RAG) ----------
EMBED_MODEL = "nomic-embed-text"          # run once: ollama pull nomic-embed-text
# nomic-embed-text was trained with these prefixes. If you switch to another
# embedding model, set both to "" unless its model card says otherwise.
EMBED_DOC_PREFIX = "search_document: "
EMBED_QUERY_PREFIX = "search_query: "

CHROMA_DIR = "data/chroma"                # vector database folder (git-ignored with data/)
CHUNK_SIZE = 900                          # characters per chunk
CHUNK_OVERLAP = 150                       # characters shared between neighbouring chunks
MAX_CHUNKS_PER_FILE = 3000                # refuse absurdly large files
TOP_K = 4                                 # passages returned per search
MAX_DISTANCE = 0.75                       # cosine distance; larger = less similar, dropped


def get_llm():
    return ChatOllama(model=MODEL_NAME, temperature=0, num_ctx=NUM_CTX)


def get_embeddings():
    from langchain_ollama import OllamaEmbeddings
    return OllamaEmbeddings(model=EMBED_MODEL)