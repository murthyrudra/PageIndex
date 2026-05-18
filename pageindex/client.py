import os
import uuid
import json
import asyncio
import concurrent.futures
from pathlib import Path
import numpy as np
from tqdm import tqdm
import PyPDF2
from openai import OpenAI

from .page_index import page_index
from .page_index_md import md_to_tree
from .retrieve import (
    get_document,
    get_document_structure,
    get_page_content,
    get_node_content,
)
from .utils import ConfigLoader, remove_fields

META_INDEX = "_meta.json"


def _normalize_retrieve_model(model: str) -> str:
    """Preserve supported Agents SDK prefixes and route other provider paths via LiteLLM."""
    passthrough_prefixes = ("litellm/", "openai/", "rits/")
    if not model or "/" not in model:
        return model
    if model.startswith(passthrough_prefixes):
        return model
    return f"litellm/{model}"


class PageIndexClient:
    """
    A client for indexing and retrieving document content.
    Flow: index() -> get_document() / get_document_structure() / get_page_content()

    For agent-based QA, see examples/agentic_vectorless_rag_demo.py.
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        retrieve_model: str = None,
        retrieval_model_url: str = None,
        workspace: str = None,
        rits_api_key: str = None,
    ):
        # Handle API key configuration
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        elif not os.getenv("OPENAI_API_KEY") and os.getenv("CHATGPT_API_KEY"):
            os.environ["OPENAI_API_KEY"] = os.getenv("CHATGPT_API_KEY")

        # Handle RITS API key configuration
        if rits_api_key:
            os.environ["RITS_API_KEY"] = rits_api_key
        elif not os.getenv("RITS_API_KEY") and os.getenv("OPENAI_API_KEY"):
            # If no RITS key but OpenAI key exists, models will use OpenAI key
            pass

        self.rits_api_key = rits_api_key

        self.workspace = Path(workspace).expanduser() if workspace else None
        overrides = {}
        if model:
            overrides["model"] = model
        if retrieve_model:
            overrides["retrieve_model"] = retrieve_model
        opt = ConfigLoader().load(overrides or None)
        self.model = opt.model
        self.retrieve_model = opt.retrieve_model
        self.retrieve_model_url = retrieval_model_url
        if self.workspace:
            self.workspace.mkdir(parents=True, exist_ok=True)
        self.documents = {}
        if self.workspace:
            self._load_workspace()

        self.embedding_client = OpenAI(
            api_key="dummy",
            base_url=self.retrieve_model_url,
        )

        self.doc_embeddings = {}

    def index(self, file_path: str, mode: str = "auto") -> str:
        """Index a document. Returns a document_id."""
        # Persist a canonical absolute path so workspace reloads do not
        # reinterpret caller-relative paths against the workspace directory.
        file_path = os.path.abspath(os.path.expanduser(file_path))
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        doc_id = str(uuid.uuid4())
        ext = os.path.splitext(file_path)[1].lower()

        is_pdf = ext == ".pdf"
        is_md = ext in [".md", ".markdown"]

        if mode == "pdf" or (mode == "auto" and is_pdf):
            print(f"Indexing PDF: {file_path}")
            result = page_index(
                doc=file_path,
                model=self.model,
                if_add_node_summary="yes",
                if_add_node_text="yes",
                if_add_node_id="yes",
                if_add_doc_description="yes",
            )
            # Extract per-page text so queries don't need the original PDF
            pages = []
            with open(file_path, "rb") as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for i, page in enumerate(pdf_reader.pages, 1):
                    pages.append({"page": i, "content": page.extract_text() or ""})

            self.documents[doc_id] = {
                "id": doc_id,
                "type": "pdf",
                "path": file_path,
                "doc_name": result.get("doc_name", ""),
                "doc_description": result.get("doc_description", ""),
                "page_count": len(pages),
                "structure": result["structure"],
                "pages": pages,
            }

        elif mode == "md" or (mode == "auto" and is_md):
            print(f"Indexing Markdown: {file_path}")
            coro = md_to_tree(
                md_path=file_path,
                if_thinning=False,
                if_add_node_summary="no",
                summary_token_threshold=200,
                model=self.model,
                if_add_doc_description="no",
                if_add_node_text="yes",
                if_add_node_id="yes",
            )
            try:
                asyncio.get_running_loop()
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    result = pool.submit(asyncio.run, coro).result()
            except RuntimeError:
                result = asyncio.run(coro)
            self.documents[doc_id] = {
                "id": doc_id,
                "type": "md",
                "path": file_path,
                "doc_name": result.get("doc_name", ""),
                "doc_description": result.get("doc_description", ""),
                "line_count": result.get("line_count", 0),
                "structure": result["structure"],
            }
        else:
            raise ValueError(f"Unsupported file format for: {file_path}")

        print(f"Indexing complete. Document ID: {doc_id}")
        if self.workspace:
            self._save_doc(doc_id)
        return doc_id

    @staticmethod
    def _make_meta_entry(doc: dict) -> dict:
        """Build a lightweight meta entry from a document dict."""
        entry = {
            "type": doc.get("type", ""),
            "doc_name": doc.get("doc_name", ""),
            "doc_description": doc.get("doc_description", ""),
            "path": doc.get("path", ""),
        }
        if doc.get("type") == "pdf":
            entry["page_count"] = doc.get("page_count")
        elif doc.get("type") == "md":
            entry["line_count"] = doc.get("line_count")
        return entry

    @staticmethod
    def _read_json(path) -> dict | None:
        """Read a JSON file, returning None on any error."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: corrupt {Path(path).name}: {e}")
            return None

    def _save_doc(self, doc_id: str):
        doc = self.documents[doc_id].copy()
        # Strip text from structure nodes — redundant with pages (PDF only)
        if doc.get("structure") and doc.get("type") == "pdf":
            doc["structure"] = remove_fields(doc["structure"], fields=["text"])
        path = self.workspace / f"{doc_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
        self._save_meta(doc_id, self._make_meta_entry(doc))
        # Drop heavy fields; will lazy-load on demand
        self.documents[doc_id].pop("structure", None)
        self.documents[doc_id].pop("pages", None)

    def _rebuild_meta(self) -> dict:
        """Scan individual doc JSON files and return a meta dict."""
        meta = {}
        for path in self.workspace.glob("*.json"):
            if path.name == META_INDEX:
                continue
            doc = self._read_json(path)
            if doc and isinstance(doc, dict):
                meta[path.stem] = self._make_meta_entry(doc)
        return meta

    def _read_meta(self) -> dict | None:
        """Read and validate _meta.json, returning None on any corruption."""
        meta = self._read_json(self.workspace / META_INDEX)
        if meta is not None and not isinstance(meta, dict):
            print(f"Warning: {META_INDEX} is not a JSON object, ignoring")
            return None
        return meta

    def _save_meta(self, doc_id: str, entry: dict):
        meta = self._read_meta() or self._rebuild_meta()
        meta[doc_id] = entry
        meta_path = self.workspace / META_INDEX
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def _load_workspace(self):
        meta = self._read_meta()
        if meta is None:
            meta = self._rebuild_meta()
            if meta:
                print(f"Loaded {len(meta)} document(s) from workspace (legacy mode).")
        for doc_id, entry in meta.items():
            doc = dict(entry, id=doc_id)
            if doc.get("path") and not os.path.isabs(doc["path"]):
                doc["path"] = str((self.workspace / doc["path"]).resolve())
            self.documents[doc_id] = doc

    def _ensure_doc_loaded(self, doc_id: str):
        """Load full document JSON on demand (structure, pages, etc.)."""
        doc = self.documents.get(doc_id)
        if not doc or doc.get("structure") is not None:
            return
        full = self._read_json(self.workspace / f"{doc_id}.json")
        if not full:
            return
        doc["structure"] = full.get("structure", [])
        if full.get("pages"):
            doc["pages"] = full["pages"]

    def get_document(self, doc_id: str) -> str:
        """Return document metadata JSON."""
        return get_document(self.documents, doc_id)

    def get_document_structure(self, doc_id: str) -> str:
        """Return document tree structure JSON (without text fields)."""
        if self.workspace:
            self._ensure_doc_loaded(doc_id)
        return get_document_structure(self.documents, doc_id)

    def get_page_content(self, doc_id: str, pages: str) -> str:
        """Return page content for the given pages string (e.g. '5-7', '3,8', '12')."""
        if self.workspace:
            self._ensure_doc_loaded(doc_id)
        return get_page_content(self.documents, doc_id, pages)

    def get_node_content(self, doc_id: str, nodes: str) -> str:
        """Return node content for the given nodes string (e.g. '5-7', '3,8', '12')."""
        if self.workspace:
            self._ensure_doc_loaded(doc_id)
        return get_node_content(self.documents, doc_id, nodes)

    def _embed(self, text: str) -> np.ndarray:
        """
        Generate dense embedding for text.
        """

        response = self.embedding_client.embeddings.create(
            model=self.retrieve_model,
            input=text,
            extra_headers={"RITS_API_KEY": self.rits_api_key},
        )

        embedding = response.data[0].embedding
        return np.array(embedding, dtype=np.float32)

    def _cosine_similarity(
        self,
        a: np.ndarray,
        b: np.ndarray,
    ) -> float:
        """
        Cosine similarity between two vectors.
        """

        denom = np.linalg.norm(a) * np.linalg.norm(b)

        if denom == 0:
            return 0.0

        return float(np.dot(a, b) / denom)

    def build_dense_index(self):
        """
        Build dense index only if it does not already exist.
        Saves embeddings inside:
            {workspace}/dense_index/
        """

        index_dir = Path(self.workspace) / "dense_index"

        embeddings_file = index_dir / "embeddings.npy"
        doc_ids_file = index_dir / "doc_ids.json"

        # ---------------------------------------------------
        # Load existing index
        # ---------------------------------------------------
        if embeddings_file.exists() and doc_ids_file.exists():

            print("Loading existing dense index...")

            embeddings = np.load(embeddings_file)

            with open(doc_ids_file, "r", encoding="utf-8") as f:
                doc_ids = json.load(f)

            self.doc_embeddings = {
                doc_id: embeddings[i] for i, doc_id in enumerate(doc_ids)
            }

            print(f"Loaded dense index with " f"{len(self.doc_embeddings)} documents.")

            return

        # ---------------------------------------------------
        # Build new index
        # ---------------------------------------------------
        print("Building dense index...")

        index_dir.mkdir(parents=True, exist_ok=True)

        self.doc_embeddings = {}

        all_embeddings = []
        all_doc_ids = []

        for doc_id, doc_data in tqdm(self.documents.items()):

            text = doc_data["doc_description"]

            embedding = self._embed(text)

            self.doc_embeddings[doc_id] = embedding

            all_embeddings.append(embedding)
            all_doc_ids.append(doc_id)

        # Convert to matrix
        embedding_matrix = np.stack(all_embeddings)

        # Save to disk
        np.save(embeddings_file, embedding_matrix)

        with open(doc_ids_file, "w", encoding="utf-8") as f:
            json.dump(all_doc_ids, f, ensure_ascii=False, indent=2)

        print(f"Built and saved dense index for " f"{len(all_doc_ids)} documents.")

    def search(
        self,
        query: str,
        top_k: int = 10,
    ):
        """
        Dense semantic retrieval.
        Returns top-k most similar documents.
        """

        query_embedding = self._embed(query)

        scores = []

        for doc_id, doc_embedding in self.doc_embeddings.items():

            score = self._cosine_similarity(
                query_embedding,
                doc_embedding,
            )

            scores.append(
                {
                    "doc_id": doc_id,
                    "score": score,
                }
            )

        scores.sort(
            key=lambda x: x["score"],
            reverse=True,
        )

        return scores[:top_k]
