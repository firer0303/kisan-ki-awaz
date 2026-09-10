"""
Kisan Ki Awaz - RAG Knowledge Base Loader
==========================================
Loads and indexes agricultural knowledge documents for
retrieval-augmented generation.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger


class KnowledgeDocument:
    """Represents a single document in the agricultural knowledge base."""

    def __init__(self, data: Dict):
        self.id: str = data.get("id", "")
        self.title: str = data.get("title", "")
        self.organization: str = data.get("organization", "")
        self.document_title: str = data.get("document_title", "")
        self.publication_date: str = data.get("publication_date", "")
        self.url: str = data.get("url", "")
        self.crop: str = data.get("crop", "")
        self.topic: str = data.get("topic", "")
        self.category: str = data.get("category", "")
        self.source_credibility: str = data.get("source_credibility", "unknown")
        self.content: str = data.get("content", "")
        self.keywords: List[str] = data.get("keywords", [])

    def to_retrieval_text(self) -> str:
        """Format document for inclusion in LLM context."""
        return (
            f"[Source: {self.organization} | {self.document_title} | "
            f"Published: {self.publication_date}]\n"
            f"Topic: {self.title}\n"
            f"Content: {self.content}\n"
            f"Reference: {self.url}\n"
            f"Credibility: {self.source_credibility}"
        )

    def to_source_citation(self) -> Dict:
        """Return a citation dict for UI display."""
        return {
            "id": self.id,
            "organization": self.organization,
            "document_title": self.document_title,
            "publication_date": self.publication_date,
            "url": self.url,
            "credibility": self.source_credibility,
        }


class KnowledgeBaseLoader:
    """
    Loads all JSON knowledge base files from the data/knowledge_base directory
    and provides in-memory retrieval.
    """

    def __init__(self, kb_dir: str = "data/knowledge_base"):
        self.kb_dir = Path(kb_dir)
        self.documents: List[KnowledgeDocument] = []
        self._load_all()

    def _load_all(self) -> None:
        """Load all JSON files from the knowledge base directory."""
        if not self.kb_dir.exists():
            logger.warning(f"Knowledge base directory not found: {self.kb_dir}")
            return

        json_files = list(self.kb_dir.glob("*.json"))
        if not json_files:
            logger.warning(f"No JSON files found in {self.kb_dir}")
            return

        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        self.documents.append(KnowledgeDocument(item))
                logger.info(
                    f"Loaded {len(data) if isinstance(data, list) else 1} "
                    f"documents from {json_file.name}"
                )
            except Exception as e:
                logger.error(f"Failed to load {json_file.name}: {e}")

        logger.info(f"Total knowledge base documents loaded: {len(self.documents)}")

    def search(
        self,
        query: str,
        category: Optional[str] = None,
        crop: Optional[str] = None,
        top_k: int = 5,
    ) -> List[KnowledgeDocument]:
        """
        Simple keyword-based search across the knowledge base.

        For production, this should be replaced with vector similarity search
        using sentence-transformers + FAISS (see rag_service.py for the
        embedding-based retrieval layer).
        """
        query_lower = query.lower()
        query_tokens = set(query_lower.split())
        scored: List[tuple] = []

        for doc in self.documents:
            # Apply category/crop filters if specified
            if category and doc.category != category:
                continue
            if crop and doc.crop != crop and doc.crop != "all_crops":
                continue

            # Calculate relevance score
            score = 0.0

            # Check title
            if query_lower in doc.title.lower():
                score += 5.0

            # Check keywords (high weight)
            for kw in doc.keywords:
                if kw.lower() in query_lower:
                    score += 3.0

            # Check content overlap
            content_lower = doc.content.lower()
            for token in query_tokens:
                if len(token) > 3 and token in content_lower:
                    score += 1.0

            # Boost authoritative sources
            if doc.source_credibility == "authoritative":
                score += 1.0

            if score > 0:
                scored.append((score, doc))

        # Sort by score descending, return top_k
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]

    def get_by_id(self, doc_id: str) -> Optional[KnowledgeDocument]:
        """Retrieve a document by its unique ID."""
        for doc in self.documents:
            if doc.id == doc_id:
                return doc
        return None

    def get_all_crops(self) -> List[str]:
        """Return a list of all unique crops in the knowledge base."""
        crops = set()
        for doc in self.documents:
            crops.add(doc.crop)
        return sorted(crops)

    def get_all_categories(self) -> List[str]:
        """Return a list of all unique categories."""
        categories = set()
        for doc in self.documents:
            categories.add(doc.category)
        return sorted(categories)

    @property
    def total_documents(self) -> int:
        return len(self.documents)
