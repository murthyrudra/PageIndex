#!/usr/bin/env python3
"""
Vectorless RAG using PageIndex Tree Structure
Standalone program for question answering using document tree structure.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm, trange

# Import litellm for LLM calls
import litellm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure litellm
litellm.drop_params = True

# Set up RITS API if available
if os.getenv("RITS_API_KEY"):
    os.environ["RITS_API_KEY"] = os.getenv("RITS_API_KEY")
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.getenv("RITS_API_KEY")


def llm_completion(model, prompt):
    """
    Call LLM using litellm with RITS support.
    """
    if model:
        model = model.removeprefix("litellm/")

    max_retries = 3
    messages = [{"role": "user", "content": prompt}]

    # Prepare kwargs for litellm.completion
    completion_kwargs = {
        "model": model,
        "messages": messages,
        "temperature": 0,
    }

    # Add RITS-specific configuration if using RITS models
    if model and (
        "rits" in model.lower()
        or model.startswith("rits/")
        or "sarvam" in model.lower()
    ):
        rits_api_base = os.getenv("RITS_API_BASE")
        rits_api_key = os.getenv("RITS_API_KEY")

        if rits_api_base:
            completion_kwargs["api_base"] = rits_api_base

        if rits_api_key:
            completion_kwargs["api_key"] = rits_api_key
            completion_kwargs["extra_headers"] = {
                "RITS_API_KEY": rits_api_key,
                "reasoning_effort": "high",
            }

    for i in range(max_retries):
        try:
            response = litellm.completion(**completion_kwargs)
            return response.choices[0].message.content
        except Exception as e:
            print(f"⚠️  Retry {i+1}/{max_retries}: {e}")
            if i < max_retries - 1:
                import time

                time.sleep(1)
            else:
                raise Exception(f"Max retries reached: {e}")


def extract_json(content):
    """Extract JSON from LLM response."""
    try:
        # First, try to extract JSON enclosed within ```json and ```
        start_idx = content.find("```json")
        if start_idx != -1:
            start_idx += 7
            end_idx = content.rfind("```")
            json_content = content[start_idx:end_idx].strip()
        else:
            json_content = content.strip()

        # Clean up common issues
        json_content = json_content.replace("None", "null")
        json_content = json_content.replace("\n", " ").replace("\r", " ")
        json_content = " ".join(json_content.split())

        return json.loads(json_content)
    except json.JSONDecodeError as e:
        print(f"⚠️  Failed to parse JSON: {e}")
        try:
            json_content = json_content.replace(",]", "]").replace(",}", "}")
            return json.loads(json_content)
        except:
            return {}


def load_tree_structure(tree_path):
    """Load precomputed tree structure from JSON file."""
    if not os.path.exists(tree_path):
        raise FileNotFoundError(f"Tree file not found: {tree_path}")

    with open(tree_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle both formats: direct tree or wrapped in structure key
    if isinstance(data, dict) and "structure" in data:
        tree = data["structure"]
        doc_name = data.get("doc_name", "Unknown")
        print(f"✓ Loaded tree for document: {doc_name}")
    else:
        tree = data
        print(f"✓ Loaded tree structure")

    return tree


def find_nodes_by_ids(tree, target_ids):
    """Recursively walk the tree and collect nodes matching target_ids."""
    found = []
    for node in tree:
        if node.get("node_id") in target_ids:
            found.append(node)
        if node.get("nodes"):
            found.extend(find_nodes_by_ids(node["nodes"], target_ids))
    return found


def evaluate_nodes_batch(query, nodes, model="gpt-4o"):
    """
    Evaluate a batch of nodes for relevance to the query.

    Args:
        query: User's question
        nodes: List of node dicts with node_id, title, summary
        model: LLM model to use

    Returns: list of node_ids that are relevant
    """
    if not nodes:
        return []

    # Build compact representation of nodes
    nodes_info = []
    for node in nodes:
        node_info = {
            "node_id": node.get("node_id"),
            "title": node.get("title"),
            "line_num": node.get("line_num", "?"),
            "summary": (
                node.get("summary")
                or node.get("prefix_summary")
                or node.get("text", "")
            )[:200],
        }
        nodes_info.append(node_info)

    prompt = f"""You are evaluating which document sections are relevant to answer a query.

Query: {query}

Sections to evaluate:
{json.dumps(nodes_info, indent=2, ensure_ascii=False)}

For each section, determine if it's relevant to answering the query.
Reply ONLY in this exact JSON format:
{{
  "relevant_node_ids": ["node_id1", "node_id2"]
}}

Include only the node_ids of sections that are relevant. If none are relevant, return an empty list."""

    response = llm_completion(model=model, prompt=prompt)
    if response is None:
        return []

    # Handle <think> tags from RITS models
    if "</think>" in response:
        response = response.split("</think>")[-1].strip()

    # Extract JSON from response
    result = extract_json(response)
    return result.get("relevant_node_ids", [])


def llm_tree_search(query, tree, model="gpt-4o", max_results=5, batch_size=40):
    """
    Core PageIndex retrieval using batch node evaluation.
    Evaluates k nodes at a time to balance efficiency and token usage.

    Args:
        query: User's question
        tree: Document tree structure
        model: LLM model to use
        max_results: Maximum number of relevant nodes to return
        batch_size: Number of nodes to evaluate in each batch (default: 5)

    Returns: dict with 'thinking' (reasoning) and 'node_list' (node IDs)
    """
    relevant_nodes = []
    total_nodes = 0
    all_nodes = []

    def collect_all_nodes(nodes, depth=0):
        """Flatten tree into list of (node, depth) tuples."""
        for node in nodes:
            all_nodes.append((node, depth))
            if node.get("nodes"):
                collect_all_nodes(node["nodes"], depth + 1)

    # Collect all nodes first
    collect_all_nodes(tree)
    total_nodes = len(all_nodes)

    print(
        f"   Evaluating {total_nodes} nodes in batches of {batch_size} (max {max_results} results)..."
    )

    # Process nodes in batches
    for i in trange(0, len(all_nodes), batch_size):
        if len(relevant_nodes) >= max_results:
            break

        batch = all_nodes[i : i + batch_size]
        batch_nodes = [node for node, _ in batch]

        # Evaluate batch
        relevant_ids = evaluate_nodes_batch(query, batch_nodes, model=model)

        # Add relevant nodes with their depth info
        for node, depth in batch:
            if node.get("node_id") in relevant_ids:
                relevant_nodes.append(
                    {
                        "node_id": node.get("node_id"),
                        "title": node.get("title"),
                        "depth": depth,
                    }
                )
                print(f"   ✓ [{node.get('node_id')}] {node.get('title')}")

                if len(relevant_nodes) >= max_results:
                    break

    # Extract just the node IDs
    node_ids = [n["node_id"] for n in relevant_nodes]

    # Create reasoning summary
    thinking = (
        f"Evaluated {total_nodes} nodes in batches of {batch_size} and found {len(relevant_nodes)} relevant sections: "
        + ", ".join([f"{n['title']} (ID: {n['node_id']})" for n in relevant_nodes[:3]])
    )

    if len(relevant_nodes) > 3:
        thinking += f" and {len(relevant_nodes) - 3} more."

    return {"thinking": thinking, "node_list": node_ids}


def generate_answer(query, nodes, model="gpt-4o"):
    """
    Takes retrieved nodes as context and generates a grounded answer.
    Instructs the LLM to cite section titles and line numbers.
    """
    if not nodes:
        return "⚠️ No relevant sections found in the document."

    # Build context string from retrieved nodes
    context_parts = []
    for node in nodes:
        line_num = node.get("line_num", "?")
        context_parts.append(
            f"[Section: '{node.get('title')}' | Line {line_num}]\n"
            f"{node.get('text', node.get('summary', node.get('prefix_summary', 'Content not available.')))}"
        )
    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are an expert document analyst.
Answer the question using ONLY the provided context.
For every claim you make, cite the section title and line number in parentheses.
Be concise and precise.

Question: {query}

Context:
{context}

Answer:"""

    return llm_completion(model=model, prompt=prompt)


def vectorless_rag(query, tree, model="gpt-4o", verbose=True):
    """
    Full end-to-end PageIndex RAG pipeline:

    Step 1: LLM Tree Search  → finds relevant node_ids
    Step 2: Node Retrieval   → fetches section content
    Step 3: Answer Generation → produces cited answer
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"🔍 Query: {query}")
        print(f"{'='*70}")

    # Step 1: Tree Search
    if verbose:
        print("\n[Step 1/3] 🌳 Searching document tree...")

    search_result = llm_tree_search(query, tree, model=model)
    node_ids = search_result.get("node_list", [])

    if verbose:
        thinking = search_result.get("thinking", "")
        print(f"\n🧠 LLM Reasoning:")
        print(f"   {thinking[:300]}{'...' if len(thinking) > 300 else ''}")
        print(f"\n🎯 Retrieved node IDs: {node_ids}")

    # Step 2: Retrieve nodes
    if verbose:
        print(f"\n[Step 2/3] 📄 Retrieving sections...")

    nodes = find_nodes_by_ids(tree, node_ids)

    if verbose:
        print(f"   Found {len(nodes)} sections:")
        for node in nodes:
            print(f"   - [{node.get('node_id')}] {node.get('title')}")

    # Step 3: Generate answer
    if verbose:
        print(f"\n[Step 3/3] 💡 Generating answer...")

    answer = generate_answer(query, nodes, model=model)

    if verbose:
        print(f"\n{'='*70}")
        print(f"📝 Answer:")
        print(f"{'='*70}")
        print(answer)
        print(f"{'='*70}\n")

    return answer


def main():
    parser = argparse.ArgumentParser(
        description="Vectorless RAG using PageIndex Tree Structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python run_rag.py --tree results/document_structure.json --query "What is machine learning?"
  
  # With RITS model
  python3 run_rag.py --tree results/sarvam_output_md_orientation_corrected_structure.json --query "Explain the concept" --model "hosted_vllm/sarvamai/sarvam-105b"
  
  # Quiet mode
  python run_rag.py --tree results/doc.json --query "Summary?" --quiet
        """,
    )

    parser.add_argument(
        "--tree",
        type=str,
        required=True,
        help="Path to precomputed tree structure JSON file",
    )

    parser.add_argument(
        "--query",
        type=str,
        default="శిలీంద్రాలు (fungal spores) పురుగుల శరీరంపై ఎలా దాడి చేస్తాయి, వాటి ఉపయోగం ఏ ప్రయోజనాలు కలిగిస్తాయి?",
        help="Question to ask about the document",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="LLM model to use (default: gpt-4o). Supports RITS models like 'hosted_vllm/sarvamai/sarvam-105b'",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output, only show final answer",
    )

    args = parser.parse_args()

    try:
        # Load tree structure
        if not args.quiet:
            print(f"📂 Loading tree from: {args.tree}")

        tree = load_tree_structure(args.tree)

        if not args.quiet:
            print(f"   Tree contains {len(tree)} top-level nodes")
            print(f"🤖 Using model: {args.model}")

        # Run RAG pipeline
        answer = vectorless_rag(
            query=args.query, tree=tree, model=args.model, verbose=not args.quiet
        )

        if args.quiet:
            print(answer)

        return 0

    except FileNotFoundError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

# Made with Bob
