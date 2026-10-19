"""Calls the live API (1 credit per test). Skipped unless CONTEXTFETCH_API_KEY is set."""

import os

import pytest

from llama_index.readers.contextfetch import ContextFetchReader

pytestmark = pytest.mark.skipif(
    not os.environ.get("CONTEXTFETCH_API_KEY"),
    reason="CONTEXTFETCH_API_KEY is not set",
)


def test_live_load_url():
    docs = ContextFetchReader().load_data(urls=["https://example.com/"])
    assert len(docs) == 1
    assert docs[0].metadata["url"].startswith("https://example.com")
    assert docs[0].metadata["title"] == "Example Domain"
    assert "documentation examples" in docs[0].text


def test_live_load_query():
    docs = ContextFetchReader().load_data(
        query="python asyncio task groups", num_results=2
    )
    assert 1 <= len(docs) <= 2
    assert all(d.metadata["url"].startswith("http") for d in docs)
