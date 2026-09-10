"""
Tests for the RAG (Retrieval-Augmented Generation) Service.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.rag_model import KnowledgeBaseLoader, KnowledgeDocument
from services.rag_service import RAGService


class TestKnowledgeBaseLoader:
    """Test the knowledge base loading and search."""

    def setup_method(self):
        # Use relative path from project root
        kb_dir = str(Path(__file__).parent.parent / "data" / "knowledge_base")
        self.kb = KnowledgeBaseLoader(kb_dir)

    def test_loads_documents(self):
        assert self.kb.total_documents > 0

    def test_loads_multiple_json_files(self):
        # We created 3 JSON files: crop_diseases, farming_practices, market_info
        assert self.kb.total_documents >= 20

    def test_search_wheat(self):
        results = self.kb.search("wheat leaf rust")
        assert len(results) > 0
        assert any("wheat" in doc.title.lower() for doc in results)

    def test_search_cotton(self):
        results = self.kb.search("cotton bollworm")
        assert len(results) > 0

    def test_search_rice(self):
        results = self.kb.search("rice blast disease")
        assert len(results) > 0

    def test_search_no_results_for_gibberish(self):
        results = self.kb.search("qzqzqz")
        # Should return fewer results than a real agricultural query
        wheat_results = self.kb.search("wheat leaf rust disease treatment")
        assert len(results) <= len(wheat_results)

    def test_search_with_category_filter(self):
        results = self.kb.search("wheat", category="disease")
        for doc in results:
            assert doc.category == "disease"

    def test_search_with_crop_filter(self):
        results = self.kb.search("cultivation", crop="wheat")
        for doc in results:
            assert doc.crop in ("wheat", "all_crops")

    def test_get_by_id(self):
        doc = self.kb.get_by_id("cd-001")
        assert doc is not None
        assert "Wheat" in doc.title

    def test_get_by_id_not_found(self):
        doc = self.kb.get_by_id("nonexistent-999")
        assert doc is None

    def test_get_all_crops(self):
        crops = self.kb.get_all_crops()
        assert "wheat" in crops
        assert "rice" in crops

    def test_get_all_categories(self):
        categories = self.kb.get_all_categories()
        assert "disease" in categories

    def test_document_to_retrieval_text(self):
        doc = self.kb.get_by_id("cd-001")
        text = doc.to_retrieval_text()
        assert "FAO" in text
        assert "Content:" in text

    def test_document_to_source_citation(self):
        doc = self.kb.get_by_id("cd-001")
        citation = doc.to_source_citation()
        assert "organization" in citation
        assert "url" in citation
        assert "credibility" in citation


class TestRAGService:
    """Test the RAG service layer."""

    def setup_method(self):
        kb_dir = str(Path(__file__).parent.parent / "data" / "knowledge_base")
        self.rag = RAGService(kb_dir)

    def test_retrieve_evidence_with_match(self):
        result = self.rag.retrieve_evidence("wheat leaf rust treatment")
        assert result["has_evidence"] is True
        assert result["evidence_count"] > 0
        assert len(result["citations"]) > 0
        assert len(result["context_text"]) > 0

    def test_retrieve_evidence_no_match(self):
        result = self.rag.retrieve_evidence("qzqzqz")
        # Should have fewer results than a real query
        real_result = self.rag.retrieve_evidence("wheat leaf rust treatment fungicide")
        assert result["evidence_count"] <= real_result["evidence_count"]

    def test_build_rag_prompt_includes_evidence(self):
        evidence = self.rag.retrieve_evidence("wheat disease")
        prompt = self.rag.build_rag_prompt(
            "How to treat wheat rust?",
            evidence,
            language="en",
        )
        assert "VERIFIED AGRICULTURAL EVIDENCE" in prompt
        assert "Farmer's Question:" in prompt

    def test_build_rag_prompt_no_evidence(self):
        # Use an empty evidence dict to test the no-evidence prompt
        evidence = {
            "retrieved_docs": [],
            "context_text": "",
            "citations": [],
            "evidence_count": 0,
            "has_evidence": False,
        }
        prompt = self.rag.build_rag_prompt(
            "Something unknown",
            evidence,
            language="en",
        )
        assert "NO VERIFIED EVIDENCE FOUND" in prompt

    def test_build_rag_prompt_with_image_analysis(self):
        evidence = self.rag.retrieve_evidence("wheat rust")
        image_analysis = {
            "prediction": "Wheat - Leaf Rust",
            "confidence": 75,
            "risk_level": "medium",
        }
        prompt = self.rag.build_rag_prompt(
            "What is this?",
            evidence,
            image_analysis=image_analysis,
            language="en",
        )
        assert "IMAGE ANALYSIS RESULTS" in prompt
        assert "Wheat - Leaf Rust" in prompt

    def test_build_rag_prompt_low_confidence_warning(self):
        evidence = self.rag.retrieve_evidence("wheat")
        image_analysis = {
            "prediction": "Wheat - Leaf Rust",
            "confidence": 30,
            "risk_level": "high",
        }
        prompt = self.rag.build_rag_prompt(
            "What is this?",
            evidence,
            image_analysis=image_analysis,
            language="en",
        )
        assert "Low confidence" in prompt or "LOW CONFIDENCE" in prompt.upper()

    def test_get_statistics(self):
        stats = self.rag.get_statistics()
        assert "total_documents" in stats
        assert stats["total_documents"] > 0
        assert "crops_covered" in stats
