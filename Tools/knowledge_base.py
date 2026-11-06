"""Knowledge-base lookup tool."""

from langchain_core.tools import tool

from RAG.knowledge import search_knowledge
from Tools.cache import cached_tool_result


@tool
def search_uploaded_knowledge(query: str, category: str | None = None) -> str:
    """Search uploaded support documents when a product answer or policy needs evidence."""
    def lookup() -> str:
        matches = search_knowledge(query, category=category)
        if not matches:
            return "No uploaded documents matched the question."
        return "\n\n".join(
            f"Source: {match['filename']} (category: {match['category']})"
            + (f", page {match['page_number']}" if match["page_number"] else "")
            + f"\n{match['content']}"
            for match in matches
        )

    return cached_tool_result(
        "search_uploaded_knowledge", {"query": query, "category": category}, lookup
    )
