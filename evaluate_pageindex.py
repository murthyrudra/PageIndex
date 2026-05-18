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
You are PageIndex, a document-grounded QA assistant.

Your job is to answer questions ONLY using information retrieved from tools.
Never use prior knowledge or make assumptions.

WORKFLOW:
1. Call get_documents(query="...", top_k=10) first using the user's question as the query.
2. get_documents performs semantic dense retrieval.
3. Only use returned documents for further exploration.
4. Use get_document(doc_id="") only when you need metadata validation.
5. Use get_document_structure(doc_id="") to locate relevant sections/pages.
6. Use get_page_content(doc_id="", pages="") ONLY for the smallest relevant ranges.
7. NEVER retrieve an entire document unless explicitly requested.
8. If the retrieved documents do not contain the answer, increase the top_k value in get_documents(query="...", top_k=10)
9. Never retrieve more than 100 documents
10. If the answer is not found, say:
   "I could not find sufficient evidence in the indexed documents."

TOOL USAGE RULES:
- Before every tool call, briefly explain why you are calling it in ONE sentence.
- Prefer narrow retrieval:
  - Good: "12", "15-17", "3,8"
  - Bad: "1-100"
- Do not repeat the same tool call with identical arguments.
- Do not fetch irrelevant sections.
- Use document structure before page retrieval whenever possible.

ANSWERING RULES:
- Answer ONLY from retrieved evidence.
- Be concise and factual.
- Cite supporting page ranges when available.
- If evidence is ambiguous, explicitly mention uncertainty.
- Do not expose internal reasoning.
- Do not mention tools unless necessary.

MULTI-DOCUMENT QUESTIONS:
- Compare evidence across documents carefully.
- Mention which document supports each claim.
- If documents conflict, state the conflict clearly.

OUTPUT STYLE:
- Short direct answer first.
- Then provide concise supporting evidence.
- Use bullet points when useful.
"""


def query_agent(
    client: PageIndexClient,
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
    def get_documents(query: str, top_k: int = 100) -> str:
        """
        Perform dense semantic search over indexed documents and
        return metadata for the most relevant documents.

        Args:
            query: Natural language search query.
            top_k: Number of documents to retrieve.
        """

        print(f"[Dense Search] query={query} with topk={top_k}")

        try:
            # Dense semantic retrieval
            search_results = client.search(
                query=query,
                top_k=top_k,
            )

            if not search_results:
                return "No relevant documents found."

            retrieved_docs = []

            for result in search_results:
                # Adjust based on your actual PageIndex result schema
                doc_id = result["doc_id"] if isinstance(result, dict) else result.doc_id

                score = (
                    result.get("score", None)
                    if isinstance(result, dict)
                    else getattr(result, "score", None)
                )

                metadata = client.get_document(doc_id)

                if score is not None:
                    retrieved_docs.append(f"[Score: {score:.4f}]\n{metadata}")
                else:
                    retrieved_docs.append(metadata)

            return "\n\n".join(retrieved_docs)

        except Exception as e:
            return f"Dense retrieval failed: {str(e)}"

    @function_tool
    def get_document(doc_id) -> str:
        """Get document metadata: status, page count, name, and description."""
        return client.get_document(doc_id)

    @function_tool
    def get_document_structure(doc_id) -> str:
        """Get the document's full tree structure (without text) to find relevant sections."""
        return client.get_document_structure(doc_id)

    @function_tool
    def get_page_content(doc_id, pages: str) -> str:
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
            get_documents,
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
        retrieve_model=completion_kwargs["RITS_EMBEDDING_MODEL"],
        retrieval_model_url=completion_kwargs["RITS_EMBEDDING_MODEL_URL"],
        workspace=str(WORKSPACE),
    )

    # Step 1: Index PDF and view tree structure
    corpus_path = f"../HippoRAG/reproduce/{args.filepath}_corpus.json"

    with open(corpus_path, "r", errors="ignore", encoding="utf8") as reader:
        documents = json.load(reader)

    os.makedirs("temp", exist_ok=True)

    test = True

    if not test:
        doc_ids = []

        for each_doc in tqdm(documents):
            title = safe_filename(each_doc["title"])
            text = each_doc["text"]

            with open(
                f"temp/{title}.md", "w", errors="ignore", encoding="utf8"
            ) as writer:
                writer.write(f"# {title}\n")
                writer.write(f"{text}")
                writer.close()

            doc_id = pageindex_client.index(f"temp/{title}.md")
            doc_ids.append(doc_id)
    else:
        pageindex_client._load_workspace()

        pageindex_client.build_dense_index()

        doc_id = list(pageindex_client.documents.keys())

        # Step 2: Evaluate on Test Set
        TEST_FILE = f"../HippoRAG/reproduce/{args.filepath}.json"
        test_instances = []
        with open(TEST_FILE, "r", encoding="utf-8") as f:
            samples = json.load(f)

        for each_sample in samples:
            instance = {}
            instance["question"] = each_sample["question"]
            instance["answer"] = each_sample["answer"]
            test_instances.append(instance)

        results = []
        for each_instance in tqdm(test_instances):
            final_output = query_agent(
                pageindex_client,
                each_instance["question"],
                verbose=True,
                model_name=model_name,
                completion_kwargs=completion_kwargs,
            )

            temp = {}
            temp["question"] = each_instance["question"]
            temp["answer"] = each_instance["answer"]
            temp["generated_answer"] = final_output
            results.append(temp)
            print(temp)

        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)

        # Save results to the specified output directory
        output_file = os.path.join(args.output_dir, "page_index_answers.json")
        with open(
            output_file,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                results,
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(f"\nResults saved to: {output_file}")
