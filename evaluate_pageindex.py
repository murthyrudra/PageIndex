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

from typing import Any, List
import sys
import re
import json
import asyncio
import concurrent.futures
from pathlib import Path
import requests
from tqdm import tqdm
import os
import argparse

from agents import Agent, Runner, function_tool, set_tracing_disabled
from agents.model_settings import ModelSettings
from agents.stream_events import RawResponsesStreamEvent, RunItemStreamEvent
from openai.types.responses import (
    ResponseTextDeltaEvent,
    ResponseReasoningSummaryTextDeltaEvent,
)

from pageindex.client import PageIndexClient
import pageindex.utils as utils


def safe_filename(name: str) -> str:
    # replace filesystem-dangerous chars
    name = re.sub(r'[\\/*?:"<>|]', "_", name)

    # optional: collapse spaces
    name = re.sub(r"\s+", " ", name).strip()

    return name


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
    doc_ids: List[str],
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
    def get_document(doc_id) -> str:
        """Get document metadata: status, page count, name, and description."""
        return client.get_document(doc_id)

    @function_tool
    def get_document_structure(doc_id) -> str:
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
    def list_documents() -> str:
        """
        Return all indexed documents with descriptions.
        """

        docs = client.list_documents()

        results = []

        for d in docs:
            results.append(
                {
                    "doc_id": d["doc_id"],
                    "title": d.get("title", ""),
                    "description": d.get("description", ""),
                }
            )

        return json.dumps(results, indent=2)

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
                    print(raw)
                elif item.type == "tool_call_output_item" and verbose:
                    output = str(item.output)
                    preview = output
                    current_stream_kind = None
                    print(output[:200])

        return streamed_run.final_output

    try:
        asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, _run()).result()
    except RuntimeError:
        return asyncio.run(_run())


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate PageIndex with document QA using command line arguments"
    )
    parser.add_argument(
        "--filepath",
        type=str,
        default="2wikimultihopqa",
        help="Path to the input file (JSON or JSONL format)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save the output results",
    )
    args = parser.parse_args()

    set_tracing_disabled(True)

    output_dir = f"{args.output_dir}/{args.filepath}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("temp", exist_ok=True)

    # Setup
    completion_kwargs, model_name = _setup_llm_key(Path("."))

    pageindex_client: Any | PageIndexClient = PageIndexClient(
        api_key="dummy",  # vLLM usually ignores this
        rits_api_key=completion_kwargs["RITS_API_KEY"],
        model=model_name.split("hosted_vllm/")[-1],
        workspace=str(WORKSPACE),
    )

    # Step 1: Index PDF and view tree structure
    corpus_path = f"/Users/rudramurthy/Documents/GitHub/HippoRAG/reproduce/{args.filepath}_corpus.json"

    with open(corpus_path, "r", errors="ignore", encoding="utf8") as reader:
        documents = json.load(reader)

    os.makedirs("temp", exist_ok=True)

    doc_ids = []

    for each_doc in tqdm(documents):
        title = safe_filename(each_doc["title"])
        text = each_doc["text"]

        with open(f"temp/{title}.md", "w", errors="ignore", encoding="utf8") as writer:
            writer.write(f"# {title}\n")
            writer.write(f"{text}")
            writer.close()

        doc_id = pageindex_client.index(f"temp/{title}.md")
        doc_ids.append(doc_id)

    # Step 2: Evaluate on Test Set
    TEST_FILE = (
        f"/Users/rudramurthy/Documents/GitHub/HippoRAG/reproduce/{args.filepath}.json"
    )
    test_instances = []
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        samples = json.load(f)

    for each_sample in samples:
        instance = {}
        instance["question"] = each_sample["question"]
        instance["answer"] = each_sample["answer"]
        test_instances.append(instance)

    # results = []
    # for each_instance in tqdm(test_instances):
    #     final_output = query_agent(
    #         pageindex_client,
    #         doc_id,
    #         each_instance["question"],
    #         verbose=True,
    #         model_name=model_name,
    #         completion_kwargs=completion_kwargs,
    #     )

    #     temp = {}
    #     temp["question"] = each_instance["question"]
    #     temp["answer"] = each_instance["answer"]
    #     each_instance["generated_answer"] = final_output
    #     results.append(temp)

    # # Create output directory if it doesn't exist
    # os.makedirs(args.output_dir, exist_ok=True)

    # # Save results to the specified output directory
    # output_file = os.path.join(args.output_dir, "page_index_answers.json")
    # with open(
    #     output_file,
    #     "w",
    #     encoding="utf-8",
    # ) as f:
    #     json.dump(
    #         results,
    #         f,
    #         indent=2,
    #         ensure_ascii=False,
    #     )

    # print(f"\nResults saved to: {output_file}")
