# LlamaIndex Readers Integration: ContextFetch

```bash
pip install llama-index-readers-contextfetch
```

The ContextFetch reader loads web pages as LlamaIndex Documents with the page's
Markdown as text, using the [ContextFetch](https://contextfetch.dev) API. Give it
URLs, a web search query, or both.

Pass `api_key`, or set the `CONTEXTFETCH_API_KEY` environment variable. With
neither, requests use ContextFetch's keyless tier, a small daily allowance per IP.
The `contextfetch` package can request a free starter key:
`from contextfetch import ContextFetch; ContextFetch().free_key()`.

## Usage

```python
from llama_index.core import VectorStoreIndex
from llama_index.readers.contextfetch import ContextFetchReader

reader = ContextFetchReader()  # or ContextFetchReader(api_key="...")

# Pages by URL
documents = reader.load_data(
    urls=["https://docs.python.org/3/library/asyncio-task.html"]
)

# The pages a web search finds
documents = reader.load_data(query="asyncio task groups", num_results=5)

index = VectorStoreIndex.from_documents(documents)
```

Each Document has `id_` set to its URL and metadata `url`, `title` and, when the
page provides them, `published_date` and `author`. A URL that fails to load is
skipped and logged as a warning.

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index/).
For an agent tool, see `llama-index-tools-contextfetch`.
