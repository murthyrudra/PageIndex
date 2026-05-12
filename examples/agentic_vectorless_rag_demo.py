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

The document is organized as a hierarchical tree of nodes.
Each node contains:
- a unique node_id
- a summary
- optional child nodes
- retrievable full text content

TOOL USE RULES:
1. Always begin with get_document() to confirm the document status and metadata.
2. Then call get_document_structure() to inspect the hierarchy, summaries, and relevant node_ids.
3. Use summaries and hierarchy to narrow down the most relevant node_ids before retrieving content.
4. Retrieve content only with get_node_content(node_ids="...").
5. Fetch only the minimum necessary nodes:
   - Prefer specific node_ids
   - Use small comma-separated sets
   - Avoid broad retrieval
   - Never retrieve the entire document unless explicitly requested
6. Before every tool call, briefly explain why the tool is needed in one short sentence.
7. Base answers strictly on retrieved tool output.
8. If the structure is insufficient to answer confidently, retrieve additional nearby or child nodes incrementally.
9. If the answer cannot be found in retrieved nodes, explicitly say so instead of guessing.
10. Be concise, factual, and cite relevant node_ids when useful.

Example retrieval flow:
- get_document()
- get_document_structure()
- get_node_content(node_ids="12,15,15.2")

Do not fabricate document content or node relationships.
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

    @function_tool
    def get_node_content(nodes: str) -> str:
        """
        Get the text content of specific nodes.
        Use tight ranges: e.g. '5-7' for nodes 5 to 7, '3,8' for nodes 3 and 8, '12' for page 12.
        For Markdown documents, use line numbers from the structure's line_num field.
        """
        return client.get_node_content(doc_id, nodes)

    from openai import OpenAI, AsyncOpenAI
    from agents import OpenAIChatCompletionsModel

    ### Translation Agent
    translation_client = AsyncOpenAI(
        api_key="dummy",  # vLLM usually ignores this
        base_url=completion_kwargs["RITS_API_BASE"],
        default_headers={"RITS_API_KEY": completion_kwargs["RITS_API_KEY"]},
    )

    translation_model = OpenAIChatCompletionsModel(
        model=model_name.split("hosted_vllm/")[-1],
        openai_client=translation_client,
    )

    translation_agent = Agent(
        name="wiki-query",
        instructions=translate_instructions,
        model=translation_model,
        model_settings=ModelSettings(parallel_tool_calls=False, max_tokens=128000),
    )

    async def translate_text(
        text: str,
        target_language: str,
    ) -> str:

        result = await Runner.run(
            translation_agent,
            f"""
        Translate the following text into {target_language}.

        TEXT:
        {text}
        """,
        )

        return result.final_output or text

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

    agent = Agent(
        name="PageIndex",
        instructions=AGENT_SYSTEM_PROMPT,
        tools=[
            get_document,
            get_document_structure,
            get_node_content,
            get_page_content,
        ],
        model=model,
        model_settings=ModelSettings(
            reasoning={"effort": "low", "summary": "auto"},
            max_tokens=128000,
        ),  # Uncomment to enable reasoning
    )

    async def _run():
        translated_question = await translate_text(prompt, target_language)

        streamed_run = Runner.run_streamed(agent, translated_question)
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
                elif item.type == "tool_call_output_item" and verbose:
                    output = str(item.output)
                    preview = output
                    current_stream_kind = None

        translated_answer = await translate_text(
            str(streamed_run.final_output), "Telugu"
        )
        return translated_answer

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

    doc_id = pageindex_client.index(
        "data/sarvam_output_md_orientation_corrected_translated.md"
    )

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
        final_output = query_agent(
            pageindex_client,
            doc_id,
            each_instance["question"],
            verbose=True,
            model_name=model_name,
            completion_kwargs=completion_kwargs,
        )

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
