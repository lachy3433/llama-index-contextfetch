"""ContextFetch tool spec."""

from typing import List, Optional

from contextfetch import ContextFetch, Result
from llama_index.core.tools.tool_spec.base import BaseToolSpec


class ContextFetchToolSpec(BaseToolSpec):
    """
    ContextFetch tool spec: web search and web pages as Markdown.

    Each function returns a Markdown string that names its sources by title
    and URL, so the agent can cite them or read them in full.

    Args:
        api_key: ContextFetch API key. Defaults to the CONTEXTFETCH_API_KEY
            environment variable; with neither, requests use the keyless tier.

    Example:
        .. code-block:: python

            from llama_index.tools.contextfetch import ContextFetchToolSpec
            from llama_index.core.agent.workflow import FunctionAgent
            from llama_index.llms.openai import OpenAI

            agent = FunctionAgent(
                tools=ContextFetchToolSpec().to_tool_list(),
                llm=OpenAI(model="gpt-4.1"),
            )
            print(await agent.run("How do asyncio task groups handle errors?"))

    """

    spec_functions = ["search", "read_page", "find_similar", "answer"]

    def __init__(self, api_key: Optional[str] = None) -> None:
        """Initialize with parameters."""
        self.client = ContextFetch(api_key=api_key)

    def search(
        self,
        query: str,
        num_results: int = 5,
        include_domains: Optional[List[str]] = None,
    ) -> str:
        """
        Search the web. Returns each result's title, URL and the passages relevant to the query.

        Use read_page to get a result's full text.

        Args:
            query: What to search for, in natural language.
            num_results: How many results to return, 1 to 25.
            include_domains: Only return results from these domains, e.g. ["docs.python.org"].

        """
        response = self.client.search(
            query,
            num_results=num_results,
            include_domains=include_domains,
            pruned_context=True,
        )
        return _format_results(response.get("results", []))

    def read_page(self, url: str, query: Optional[str] = None) -> str:
        """
        Read a web page as Markdown.

        Args:
            url: The page to read.
            query: Optional question. When given, only the parts of the page relevant to it are returned.

        """
        return self.client.read(url, query=query).markdown

    def find_similar(self, url: str, num_results: int = 5) -> str:
        """
        Find pages on other sites that are similar to a web page. Returns each result's title, URL and its most relevant passages.

        Args:
            url: The page to find similar pages for.
            num_results: How many results to return, 1 to 25.

        """
        response = self.client.find_similar(
            url, num_results=num_results, pruned_context=True
        )
        return _format_results(response.get("results", []))

    def answer(self, query: str) -> str:
        """
        Answer a question with passages quoted verbatim from web pages, each followed by [n], then the numbered sources.

        No text is generated; when no passage is relevant enough, says so.

        Args:
            query: The question.

        """
        response = self.client.answer(query)
        if not response.get("answer"):
            return "No passage was relevant enough to answer this question."
        sources = "\n".join(
            f"[{n}] {c.get('title') or c.get('url')}: {c.get('url')}"
            for n, c in enumerate(response.get("citations", []), start=1)
        )
        return f"{response['answer']}\n\nSources:\n{sources}"


def _format_results(results: List[Result]) -> str:
    blocks = []
    for r in results:
        lines = [f"## {r.get('title') or r.get('url')}", f"URL: {r.get('url')}"]
        if r.get("publishedDate"):
            lines.append(f"Published: {r['publishedDate']}")
        lines += ["", r.get("prunedContext") or ""]
        blocks.append("\n".join(lines).strip())
    return "\n\n".join(blocks) or "No results."
