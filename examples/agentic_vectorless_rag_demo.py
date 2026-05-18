"""
Agentic Vectorless RAG with PageIndex - Demo

A simple example of building a document QA agent with self-hosted PageIndex
and the OpenAI Agents SDK. Instead of vector similarity search and chunking,
PageIndex builds a hierarchical tree index and uses agentic LLM reasoning for
human-like, context-aware retrieval.

Agent tools:
  - get_document()           — document metadata (status, page count, etc.)
  - get_document_structure() — tree structure index of a document
  - get_page_content()       — retrieve text content of specific pages

Steps:
  1 — Index a PDF and view its tree structure index
  2 — View document metadata
  3 — Ask a question (agent reasons over the index and auto-calls tools)

Requirements: pip install openai-agents
"""

from typing import Any
import sys
import json
import asyncio
import concurrent.futures
from pathlib import Path
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import Agent, Runner, function_tool, set_tracing_disabled
from agents.model_settings import ModelSettings
from agents.stream_events import RawResponsesStreamEvent, RunItemStreamEvent
from openai.types.responses import (
    ResponseTextDeltaEvent,
    ResponseReasoningSummaryTextDeltaEvent,
)

from pageindex.client import PageIndexClient
import pageindex.utils as utils

translate_instructions = """
You are an expert multilingual agricultural translator.

Rules:
- Preserve crop names.
- Preserve mandi/market names.
- Preserve state names.
- Preserve prices and units exactly.
- Preserve percentages exactly.
- Preserve markdown formatting.
- Only translate human language text.
- Return only translated text.
"""


def _setup_llm_key(kb_dir: Path):
    """Set LiteLLM API key from LLM_API_KEY env var if present.

    Load order (override=False, so first one wins):
    1. System environment variables (already set)
    2. KB-local .env  (kb_dir/.env)
    3. Global .env    (~/.config/openkb/.env)

    Also propagates to provider-specific env vars (OPENAI_API_KEY, etc.)
    so that the Agents SDK litellm provider can pick them up.
    """
    import os
    from dotenv import dotenv_values

    env_file = os.path.join(kb_dir, ".env")

    config = dotenv_values(env_file)

    completion_kwargs = {}

    if "RITS_API_BASE" in config:
        completion_kwargs["RITS_API_BASE"] = config["RITS_API_BASE"]

    if "RITS_API_KEY" in config:
        completion_kwargs["RITS_API_KEY"] = config["RITS_API_KEY"]

    if "RITS_EMBEDDING_MODEL_URL" in config:
        completion_kwargs["RITS_EMBEDDING_MODEL_URL"] = config[
            "RITS_EMBEDDING_MODEL_URL"
        ]

    if "RITS_EMBEDDING_MODEL" in config:
        completion_kwargs["RITS_EMBEDDING_MODEL"] = config["RITS_EMBEDDING_MODEL"]

    return completion_kwargs, config["RITS_MODEL"]


_EXAMPLES_DIR = Path(__file__).parent
WORKSPACE = _EXAMPLES_DIR / "workspace"

AGENT_SYSTEM_PROMPT = """
You are PageIndex, a document QA assistant.
TOOL USE:
- Call get_document() first to confirm status and page/line count.
- Call get_document_structure() to identify relevant page ranges.
- Call get_page_content(pages="5-7") with tight ranges; never fetch the whole document.
- Before each tool call, output one short sentence explaining the reason.
Answer based only on tool output. Be concise.
"""


def query_agent(
    client: PageIndexClient,
    doc_id: str,
    prompt: str,
    verbose: bool = False,
    model_name: str = "",
    completion_kwargs: dict = {},
    target_language: str = "English",
) -> str:
    """Run a document QA agent using the OpenAI Agents SDK.

    Streams text output token-by-token and returns the full answer string.
    Tool calls are always printed; verbose=True also prints arguments and output previews.
    """

    from openai import OpenAI, AsyncOpenAI
    from agents import OpenAIChatCompletionsModel

    ### PageIndex Agent
    async_client: AsyncOpenAI = AsyncOpenAI(
        api_key="dummy",  # vLLM usually ignores this
        base_url=completion_kwargs["RITS_API_BASE"],
        default_headers={"RITS_API_KEY": completion_kwargs["RITS_API_KEY"]},
    )

    model = OpenAIChatCompletionsModel(
        model=model_name.split("hosted_vllm/")[-1],
        openai_client=async_client,
    )

    @function_tool
    def get_document() -> str:
        """Get document metadata: status, page count, name, and description."""
        return client.get_document(doc_id)

    @function_tool
    def get_document_structure() -> str:
        """Get the document's full tree structure (without text) to find relevant sections."""
        return client.get_document_structure(doc_id)

    @function_tool
    def get_page_content(pages: str) -> str:
        """
        Get the text content of specific pages or line numbers.
        Use tight ranges: e.g. '5-7' for pages 5 to 7, '3,8' for pages 3 and 8, '12' for page 12.
        For Markdown documents, use line numbers from the structure's line_num field.
        """
        return client.get_page_content(doc_id, pages)

    agent = Agent(
        name="PageIndex",
        instructions=AGENT_SYSTEM_PROMPT,
        tools=[
            get_document,
            get_document_structure,
            get_page_content,
        ],
        model=model,
        model_settings=ModelSettings(
            max_tokens=16000,
        ),  # Uncomment to enable reasoning
    )

    async def _run():

        streamed_run = Runner.run_streamed(agent, prompt)
        current_stream_kind = None
        async for event in streamed_run.stream_events():
            if isinstance(event, RawResponsesStreamEvent):
                if isinstance(event.data, ResponseReasoningSummaryTextDeltaEvent):
                    delta = event.data.delta
                    current_stream_kind = "reasoning"
                elif isinstance(event.data, ResponseTextDeltaEvent):
                    delta = event.data.delta
                    current_stream_kind = "text"
            elif isinstance(event, RunItemStreamEvent):
                item = event.item
                if item.type == "tool_call_item":
                    raw = item.raw_item
                    args = getattr(raw, "arguments", "{}")
                    args_str = f"({args})" if verbose else ""
                    current_stream_kind = None
                    # print(raw)
                elif item.type == "tool_call_output_item" and verbose:
                    output = str(item.output)
                    preview = output
                    current_stream_kind = None
                    # print(output[:200])

        return str(streamed_run.final_output)

    try:
        asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, _run()).result()
    except RuntimeError:
        return asyncio.run(_run())


if __name__ == "__main__":

    set_tracing_disabled(True)

    # Setup
    completion_kwargs, model_name = _setup_llm_key(Path("."))

    pageindex_client: Any | PageIndexClient = PageIndexClient(
        api_key="dummy",  # vLLM usually ignores this
        rits_api_key=completion_kwargs["RITS_API_KEY"],
        model=model_name.split("hosted_vllm/")[-1],
        workspace=str(WORKSPACE),
    )

    # Step 1: Index PDF and view tree structure

    doc_id = pageindex_client.index("data/sarvam_output_md_orientation_corrected.md")

    print(f"\nIndexed. doc_id: {doc_id}")
    print("\nTree Structure (top-level sections):")
    structure = json.loads(pageindex_client.get_document_structure(doc_id))
    utils.print_tree(structure)

    # Step 2: View document metadata
    print("\n" + "=" * 60)
    print("Step 2: View document metadata")
    print("=" * 60)
    doc_metadata = pageindex_client.get_document(doc_id)
    print(f"\n{doc_metadata}")

    # Step 3: Agent Query
    TEST_FILE = (
        "../IRL-Indic-RAG/data/ap-agri-summarization-qa-gpt-oss-120b/splits/test.jsonl"
    )
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_data = []
        for line in f:
            test_data.append(json.loads(line.strip()))

    results = []
    for each_instance in tqdm(test_data):
        try:
            final_output = query_agent(
                pageindex_client,
                doc_id,
                each_instance["question"],
                verbose=True,
                model_name=model_name,
                completion_kwargs=completion_kwargs,
            )
            print(final_output)
        except:
            final_output = ""

        temp = {}
        temp["question"] = each_instance["question"]
        temp["answer"] = each_instance["answer"]
        each_instance["generated_answer"] = final_output
        results.append(temp)

    with open(
        "page_index_answers.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False,
        )
