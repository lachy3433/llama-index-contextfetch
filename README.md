# LlamaIndex integrations for ContextFetch

[ContextFetch](https://contextfetch.dev) is a web search and web-page-to-Markdown API for AI agents. This repo holds
its two LlamaIndex packages, published to PyPI:

| Package | Import | Class |
| --- | --- | --- |
| [`llama-index-readers-contextfetch`](https://pypi.org/project/llama-index-readers-contextfetch/) ([source](llama-index-readers-contextfetch)) | `llama_index.readers.contextfetch` | `ContextFetchReader`: loads URLs or search results as Markdown `Document`s |
| [`llama-index-tools-contextfetch`](https://pypi.org/project/llama-index-tools-contextfetch/) ([source](llama-index-tools-contextfetch)) | `llama_index.tools.contextfetch` | `ContextFetchToolSpec`: `search`, `read_page`, `find_similar`, `answer` |

```bash
pip install llama-index-readers-contextfetch llama-index-tools-contextfetch
```

Both work without an API key on the free tier (25 requests a day per IP); set `CONTEXTFETCH_API_KEY` for more.
See each package's README for usage. MIT licensed.
