from langchain_core.tools import tool

from src.documents import store as default_store


def make_search_tool(doc_store):
    """Build the search tool for a given DocumentStore.
    The app uses the default store. A benchmark can pass its own isolated store."""

    @tool
    def search_documents(query: str) -> str:
        """Search the documents the user has added (PDF, Word, text files) for passages
        relevant to the query. Use this whenever a question is about the content of a
        document or file the user added. Write the query as a short description of the
        information you need, for example 'refund policy for annual plans'."""
        try:
            if doc_store.count() == 0:
                return "No documents have been added yet."
            hits = doc_store.search(query)
        except Exception as e:
            return f"Error searching documents: {e}"
        if not hits:
            return "No relevant passages found in the indexed documents."
        blocks = []
        for i, h in enumerate(hits, 1):
            where = f"{h['name']}, page {h['page']}" if h["page"] else h["name"]
            blocks.append(f"[{i}] {where} (similarity {h['score']})\n{h['text']}")
        return "\n\n".join(blocks)

    return search_documents


search_documents = make_search_tool(default_store)