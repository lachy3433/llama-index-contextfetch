# LlamaIndex Tools Integration: ContextFetch

```bash
pip install llama-index-tools-contextfetch
```

This tool connects agents to [ContextFetch](https://contextfetch.dev), a web
search and web page to Markdown API.

Pass `api_key`, or set the `CONTEXTFETCH_API_KEY` environment variable. With
neither, requests use ContextFetch's keyless tier, a small daily allowance per IP.
The `contextfetch` package can request a free starter key:
`from contextfetch import ContextFetch; ContextFetch().free_key()`.

## Usage

```python
# %pip install llama-index llama-index-tools-contextfetch

from llama_index.tools.contextfetch import ContextFetchToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

contextfetch_tool = ContextFetchToolSpec()  # or ContextFetchToolSpec(api_key="...")
agent = FunctionAgent(
    tools=contextfetch_tool.to_tool_list(),
    llm=OpenAI(model="gpt-4.1"),
)

print(await agent.run("How do asyncio task groups handle errors?"))
```

## Available Functions

`search`: Search the web. Returns each result's title, URL, published date when
known, and the passages relevant to the query.

`read_page`: Read a web page as Markdown. With `query`, only the parts of the page
relevant to it are returned.

`find_similar`: Find pages on other sites that are similar to a given page.

`answer`: Answer a question with passages quoted verbatim from web pages, each
followed by `[n]`, then the numbered source URLs. No text is generated.

Each function returns a Markdown string that names its sources by title and URL.
To load pages as Documents for indexing, use `llama-index-readers-contextfetch`.

This loader is designed to be used as a way to load data as a Tool in an Agent.
