from unittest.mock import MagicMock, patch

import pytest

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from llama_index.readers.contextfetch import ContextFetchReader

PAGE = {
    "id": "page-1",
    "url": "https://example.com/a",
    "title": "Page A",
    "publishedDate": "2026-02-11",
    "author": None,
    "text": "# Page A\n\nBody.",
}


def test_class():
    names_of_base_classes = [b.__name__ for b in ContextFetchReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


@patch("llama_index.readers.contextfetch.base.ContextFetch")
def test_init(mock_contextfetch):
    reader = ContextFetchReader(api_key="test-key")
    mock_contextfetch.assert_called_once_with(api_key="test-key")
    assert reader.client is mock_contextfetch.return_value


@patch("llama_index.readers.contextfetch.base.ContextFetch")
def test_init_without_key_defers_to_sdk(mock_contextfetch):
    ContextFetchReader()
    mock_contextfetch.assert_called_once_with(api_key=None)


@patch("llama_index.readers.contextfetch.base.ContextFetch")
def test_load_urls(mock_contextfetch):
    client = MagicMock()
    client.contents.return_value = {"results": [PAGE]}
    mock_contextfetch.return_value = client

    docs = ContextFetchReader(api_key="k").load_data(urls=["https://example.com/a"])

    client.contents.assert_called_once_with(["https://example.com/a"])
    client.search.assert_not_called()
    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert docs[0].text == "# Page A\n\nBody."
    assert docs[0].id_ == "https://example.com/a"
    # None values are dropped so vector stores that reject them still work.
    assert docs[0].metadata == {
        "url": "https://example.com/a",
        "title": "Page A",
        "published_date": "2026-02-11",
    }


@patch("llama_index.readers.contextfetch.base.ContextFetch")
def test_load_urls_in_batches(mock_contextfetch):
    client = MagicMock()
    client.contents.return_value = {"results": []}
    mock_contextfetch.return_value = client
    urls = [f"https://example.com/{i}" for i in range(23)]

    ContextFetchReader().load_data(urls=urls)

    assert [c.args[0] for c in client.contents.call_args_list] == [
        urls[0:10],
        urls[10:20],
        urls[20:23],
    ]


@patch("llama_index.readers.contextfetch.base.ContextFetch")
def test_failed_url_is_skipped_with_warning(mock_contextfetch, caplog):
    client = MagicMock()
    client.contents.return_value = {
        "results": [PAGE],
        "statuses": [
            {"id": "https://example.com/a", "status": "success"},
            {
                "id": "https://example.com/gone",
                "status": "error",
                "error": {"message": "HTTP 404"},
            },
        ],
    }
    mock_contextfetch.return_value = client

    docs = ContextFetchReader().load_data(
        urls=["https://example.com/a", "https://example.com/gone"]
    )

    assert [d.id_ for d in docs] == ["https://example.com/a"]
    assert "https://example.com/gone" in caplog.text
    assert "HTTP 404" in caplog.text


@patch("llama_index.readers.contextfetch.base.ContextFetch")
def test_load_query(mock_contextfetch):
    client = MagicMock()
    client.search.return_value = {"results": [PAGE, {"url": "https://example.com/b"}]}
    mock_contextfetch.return_value = client

    docs = ContextFetchReader().load_data(query="example", num_results=2)

    client.search.assert_called_once_with("example", num_results=2, text=True)
    client.contents.assert_not_called()
    assert [d.id_ for d in docs] == ["https://example.com/a", "https://example.com/b"]
    assert docs[1].text == ""
    assert docs[1].metadata == {"url": "https://example.com/b"}


@patch("llama_index.readers.contextfetch.base.ContextFetch")
def test_load_urls_and_query(mock_contextfetch):
    client = MagicMock()
    client.contents.return_value = {"results": [PAGE]}
    client.search.return_value = {"results": [{"url": "https://example.com/b"}]}
    mock_contextfetch.return_value = client

    docs = ContextFetchReader().load_data(urls=["https://example.com/a"], query="b")

    assert [d.id_ for d in docs] == ["https://example.com/a", "https://example.com/b"]


def test_load_requires_urls_or_query():
    with patch("llama_index.readers.contextfetch.base.ContextFetch"):
        with pytest.raises(ValueError):
            ContextFetchReader().load_data()
