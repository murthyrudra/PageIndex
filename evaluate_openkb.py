import json
import asyncio
from pathlib import Path
from tqdm import tqdm

# ---------------------------------------------------
# OpenKB Imports
# ---------------------------------------------------

from openkb.agent.query import run_query

# ---------------------------------------------------
# Config
# ---------------------------------------------------

KB_DIR = Path("/dccstor/indiclm/rudra/PageIndex/openkb/")

MODEL_NAME = "hosted_vllm/openai/gpt-oss-120b"

TEST_FILE = "/dccstor/indiclm/rudra/IRL-Indic-RAG/data/ap-agri-summarization-qa-gpt-oss-120b/splits/test.jsonl"

OUTPUT_FILE = "results/openkb_predictions.json"

STREAM = False

# ---------------------------------------------------
# Load Test Data
# ---------------------------------------------------

with open(TEST_FILE, "r", encoding="utf-8") as f:
    test_data = []
    for line in f:
        test_data.append(json.loads(line.strip()))

"""
Expected format:

[
  {
    "question": "...",
    "answer": "..."
  }
]
"""

# ---------------------------------------------------
# Generate Predictions
# ---------------------------------------------------

results = []


async def process_example(example, idx):

    question = example["question"]

    gold_answer = example.get("answer", "")

    print(f"\n{'=' * 80}")
    print(f"[{idx}] QUESTION")
    print(question)

    try:

        predicted_answer = await run_query(
            question=question,
            kb_dir=KB_DIR,
            model=MODEL_NAME,
            stream=STREAM,
        )

    except Exception as e:

        predicted_answer = f"ERROR: {str(e)}"

    result = {
        "id": idx,
        "question": question,
        "gold_answer": gold_answer,
        "predicted_answer": predicted_answer,
    }

    print(f"\nPREDICTED ANSWER:\n{predicted_answer[:1000]}")

    return result


async def main():

    tasks = []

    for idx, example in enumerate(test_data):

        tasks.append(process_example(example, idx))

    # Sequential (safer for OSS/vLLM)
    for task in tqdm(tasks):

        result = await task

        results.append(result)

        # Save incrementally
        with open(
            OUTPUT_FILE,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                results,
                f,
                indent=2,
                ensure_ascii=False,
            )

    print(f"\n✅ Saved predictions to {OUTPUT_FILE}")


# ---------------------------------------------------
# Run
# ---------------------------------------------------

if __name__ == "__main__":

    asyncio.run(main())
