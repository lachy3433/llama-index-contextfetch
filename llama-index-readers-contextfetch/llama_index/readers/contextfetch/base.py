"""ContextFetch reader."""

import logging
from typing import List, Optional

from contextfetch import ContextFetch, Result
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

logger = logging.getLogger(__name__)

# POST /v1/contents accepts at most this many URLs per call.
MAX_URLS_PER_CALL = 10


class ContextFetchReader(BaseReader):
    """
    Load web pages, or the pages a web search finds, as Markdown Documents.

    Args:
        api_key: ContextFetch API key. Defaults to the CONTEXTFETCH_API_KEY
            environment variable; with neither, requests use the keyless tier.

    Example:
        .. code-block:: python

            from llama_index.readers.contextfetch import ContextFetchReader

            reader = ContextFetchReader()
            documents = reader.load_data(
                urls=["https://docs.python.org/3/library/asyncio-task.html"]
            )
            documents = reader.load_data(query="asyncio task groups", num_results=5)

    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        """Initialize with parameters."""
        self.client = ContextFetch(api_key=api_key)

    def load_data(
        self,
        urls: Optional[List[str]] = None,
        query: Optional[str] = None,
        num_results: Optional[int] = None,
    ) -> List[Document]:
        """
        Load pages as Documents, one per page, with the page's Markdown as text.

        Args:
            urls: Pages to load. A URL that fails to load is skipped with a warning.
            query: A web search; the pages it finds are loaded.
            num_results: How many search results to load (the API default is 10).

        Returns:
            Documents with id_ set to the URL and metadata url, title and, when
            known, published_date and author.

        """
        if not urls and not query:
            raise ValueError("Pass urls, query, or both.")
        urls = urls or []
        results: List[Result] = []
        for i in range(0, len(urls), MAX_URLS_PER_CALL):
            contents = self.client.contents(urls[i : i + MAX_URLS_PER_CALL])
            results += contents.get("results", [])
            for status in contents.get("statuses", []):
                if status.get("status") == "error":
                    logger.warning(
                        "ContextFetch could not load %s: %s",
                        status.get("id"),
                        status.get("error"),
                    )
        if query:
            search = self.client.search(query, num_results=num_results, text=True)
            results += search.get("results", [])
        return [_to_document(result) for result in results]


def _to_document(result: Result) -> Document:
    metadata = {
        "url": result.get("url"),
        "title": result.get("title"),
        "published_date": result.get("publishedDate"),
        "author": result.get("author"),
    }
    return Document(
        id_=result.get("url"),
        text=result.get("text") or "",
        metadata={k: v for k, v in metadata.items() if v is not None},
    )
