from ddgs import DDGS
from langchain_core.tools import tool

@tool
def web_search(query: str) -> str:
    """Search the web for recent or factual information. Returns top results."""
    try:
        results = DDGS().text(query, max_results=4)
        return "\n\n".join(f"{r['title']}\n{r['body']}\n{r['href']}" for r in results) or "No results."
    except Exception as e:
        return f"Error: {e}"