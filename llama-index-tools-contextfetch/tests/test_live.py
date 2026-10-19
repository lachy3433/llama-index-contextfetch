"""Calls the live API (1 credit per test). Skipped unless CONTEXTFETCH_API_KEY is set."""

import os

import pytest

from llama_index.tools.contextfetch import ContextFetchToolSpec

pytestmark = pytest.mark.skipif(
    not os.environ.get("CONTEXTFETCH_API_KEY"),
    reason="CONTEXTFETCH_API_KEY is not set",
)


def test_live_search():
    out = ContextFetchToolSpec().search("python asyncio task groups", num_results=2)
    assert out != "No results."
    assert "URL: http" in out


def test_live_read_page():
    out = ContextFetchToolSpec().read_page("https://example.com/")
    assert "documentation examples" in out


def test_live_find_similar():
    out = ContextFetchToolSpec().find_similar(
        "https://docs.python.org/3/library/asyncio-task.html", num_results=2
    )
    assert out != "No results."
    assert "URL: http" in out


def test_live_answer():
    out = ContextFetchToolSpec().answer("How do I cancel an asyncio task?")
    assert "\n\nSources:\n[1] " in out
