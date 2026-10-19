from unittest.mock import MagicMock, patch

from contextfetch import Page
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.contextfetch import ContextFetchToolSpec

RESULTS = {
    "results": [
        {
            "url": "https://example.com/a",
            "title": "Page A",
            "publishedDate": "2026-02-11",
            "prunedContext": "Relevant passage A.",
        },
        {"url": "https://example.com/b", "title": None, "prunedContext": "Passage B."},
    ]
}


def make_tool(mock_contextfetch):
    client = MagicMock()
    mock_contextfetch.return_value = client
    return ContextFetchToolSpec(api_key="test-key"), client


def test_class():
    names_of_base_classes = [b.__name__ for b in ContextFetchToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes


def test_spec_functions():
    assert ContextFetchToolSpec.spec_functions == [
        "search",
        "read_page",
        "find_similar",
        "answer",
    ]


@patch("llama_index.tools.contextfetch.base.ContextFetch")
def test_init(mock_contextfetch):
    tool = ContextFetchToolSpec(api_key="test-key")
    mock_contextfetch.assert_called_once_with(api_key="test-key")
    assert tool.client is mock_contextfetch.return_value


@patch("llama_index.tools.contextfetch.base.ContextFetch")
def test_init_without_key_defers_to_sdk(mock_contextfetch):
    ContextFetchToolSpec()
    mock_contextfetch.assert_called_once_with(api_key=None)


@patch("llama_index.tools.contextfetch.base.ContextFetch")
def test_to_tool_list(mock_contextfetch):
    tools = ContextFetchToolSpec().to_tool_list()
    assert [t.metadata.name for t in tools] == ContextFetchToolSpec.spec_functions
    search = tools[0].metadata
    assert "Search the web" in search.description
    assert search.fn_schema_str and "include_domains" in search.fn_schema_str


@patch("llama_index.tools.contextfetch.base.ContextFetch")
def test_search(mock_contextfetch):
    tool, client = make_tool(mock_contextfetch)
    client.search.return_value = RESULTS

    out = tool.search("example", num_results=2, include_domains=["example.com"])

    client.search.assert_called_once_with(
        "example",
        num_results=2,
        include_domains=["example.com"],
        pruned_context=True,
    )
    assert out == (
        "## Page A\nURL: https://example.com/a\nPublished: 2026-02-11\n\n"
        "Relevant passage A.\n\n"
        "## https://example.com/b\nURL: https://example.com/b\n\nPassage B."
    )


@patch("llama_index.tools.contextfetch.base.ContextFetch")
def test_search_defaults_and_no_results(mock_contextfetch):
    tool, client = make_tool(mock_contextfetch)
    client.search.return_value = {"results": []}

    assert tool.search("nothing") == "No results."
    client.search.assert_called_once_with(
        "nothing", num_results=5, include_domains=None, pruned_context=True
    )


@patch("llama_index.tools.contextfetch.base.ContextFetch")
def test_agent_sees_source_urls(mock_contextfetch):
    tool, client = make_tool(mock_contextfetch)
    client.search.return_value = RESULTS

    output = tool.to_tool_list()[0].call(query="example")

    assert "https://example.com/a" in output.content
    assert "Relevant passage A." in output.content


@patch("llama_index.tools.contextfetch.base.ContextFetch")
def test_read_page(mock_contextfetch):
    tool, client = make_tool(mock_contextfetch)
    client.read.return_value = Page(url="https://example.com/a", markdown="# A")

    assert tool.read_page("https://example.com/a") == "# A"
    client.read.assert_called_once_with("https://example.com/a", query=None)

    tool.read_page("https://example.com/a", query="pricing")
    client.read.assert_called_with("https://example.com/a", query="pricing")


@patch("llama_index.tools.contextfetch.base.ContextFetch")
def test_find_similar(mock_contextfetch):
    tool, client = make_tool(mock_contextfetch)
    client.find_similar.return_value = RESULTS

    out = tool.find_similar("https://example.org/post", num_results=2)

    client.find_similar.assert_called_once_with(
        "https://example.org/post", num_results=2, pruned_context=True
    )
    assert "URL: https://example.com/a" in out
    assert "URL: https://example.com/b" in out


@patch("llama_index.tools.contextfetch.base.ContextFetch")
def test_answer(mock_contextfetch):
    tool, client = make_tool(mock_contextfetch)
    client.answer.return_value = {
        "answer": "Quoted passage. [1]\n\nAnother. [2]",
        "citations": [
            {"url": "https://example.com/a", "title": "Page A"},
            {"url": "https://example.com/b", "title": None},
        ],
    }

    out = tool.answer("question?")

    client.answer.assert_called_once_with("question?")
    assert out == (
        "Quoted passage. [1]\n\nAnother. [2]\n\nSources:\n"
        "[1] Page A: https://example.com/a\n"
        "[2] https://example.com/b: https://example.com/b"
    )


@patch("llama_index.tools.contextfetch.base.ContextFetch")
def test_answer_when_nothing_relevant(mock_contextfetch):
    tool, client = make_tool(mock_contextfetch)
    client.answer.return_value = {"answer": "", "citations": []}

    assert tool.answer("question?") == (
        "No passage was relevant enough to answer this question."
    )
