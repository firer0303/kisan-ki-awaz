"""
Kisan Ki Awaz - RAG Retrieval Service
=======================================
Orchestrates retrieval of relevant agricultural evidence from the
knowledge base and formats it for the LLM.
"""
from typing import Dict, List, Optional

from loguru import logger

from models.rag_model import KnowledgeBaseLoader, KnowledgeDocument


class RAGService:
    """
    Retrieval-Augmented Generation service.
    Retrieves relevant agricultural evidence and formats it for LLM consumption.
    """

    def __init__(self, kb_dir: str = "data/knowledge_base"):
        self.kb = KnowledgeBaseLoader(kb_dir)
        logger.info(
            f"RAG service initialized with {self.kb.total_documents} documents"
        )

    def retrieve_evidence(
        self,
        query: str,
        crop: Optional[str] = None,
        category: Optional[str] = None,
        top_k: int = 3,
    ) -> Dict:
        """
        Retrieve relevant evidence from the knowledge base.

        Returns:
            Dict with keys:
            - retrieved_docs: List[KnowledgeDocument]
            - context_text: str (formatted for LLM)
            - citations: List[Dict] (for UI display)
            - evidence_count: int
            - has_evidence: bool
        """
        docs = self.kb.search(query, category=category, crop=crop, top_k=top_k)

        if not docs:
            return {
                "retrieved_docs": [],
                "context_text": "",
                "citations": [],
                "evidence_count": 0,
                "has_evidence": False,
            }

        # Build context text for LLM
        context_parts = []
        citations = []
        for i, doc in enumerate(docs, 1):
            context_parts.append(
                f"--- Evidence {i} ---\n{doc.to_retrieval_text()}"
            )
            citations.append(doc.to_source_citation())

        context_text = "\n\n".join(context_parts)

        return {
            "retrieved_docs": docs,
            "context_text": context_text,
            "citations": citations,
            "evidence_count": len(docs),
            "has_evidence": True,
        }

    def build_rag_prompt(
        self,
        farmer_question: str,
        evidence: Dict,
        image_analysis: Optional[Dict] = None,
        language: str = "en",
    ) -> str:
        """
        Build the full RAG prompt that will be sent to the LLM.
        Combines the farmer's question, retrieved evidence, and
        any image analysis results.
        """
        system_instructions = (
            "You are Kisan Ki Awaz, an AI farming assistant for Pakistani farmers.\n"
            "CRITICAL RULES:\n"
            "1. Base your recommendations ONLY on the verified evidence provided below.\n"
            "2. Clearly distinguish between verified information (from sources), "
            "AI inference (logical deductions), and uncertain predictions.\n"
            "3. Never present a low-confidence diagnosis as a confirmed disease.\n"
            "4. If evidence is insufficient, state that clearly and recommend "
            "consulting a local agricultural extension officer.\n"
            "5. Use simple, farmer-friendly language appropriate for the selected language.\n"
            "6. Include specific, actionable recommendations (dosages, timing, methods).\n"
            "7. Always include a 'Verified Sources' section listing the sources used.\n"
        )

        # Build the evidence section
        evidence_section = ""
        if evidence.get("has_evidence"):
            evidence_section = (
                "\n=== VERIFIED AGRICULTURAL EVIDENCE ===\n"
                f"{evidence['context_text']}\n"
                "=== END EVIDENCE ===\n"
            )
        else:
            evidence_section = (
                "\n=== NO VERIFIED EVIDENCE FOUND ===\n"
                "No matching documents found in the verified knowledge base.\n"
                "Inform the farmer that you cannot provide a verified recommendation "
                "and suggest they consult a local agricultural extension officer.\n"
                "=== END EVIDENCE ===\n"
            )

        # Build image analysis section if available
        image_section = ""
        if image_analysis:
            confidence = image_analysis.get("confidence", 0)
            image_section = (
                "\n=== IMAGE ANALYSIS RESULTS ===\n"
                f"Detected: {image_analysis.get('prediction', 'Unknown')}\n"
                f"Confidence: {confidence}%\n"
                f"Risk Level: {image_analysis.get('risk_level', 'Unknown')}\n"
            )
            if confidence < 50:
                image_section += (
                    "WARNING: Low confidence detection. Clearly state this is a "
                    "preliminary observation and NOT a confirmed diagnosis. "
                    "Recommend the farmer consult an expert for confirmation.\n"
                )
            image_section += "=== END IMAGE ANALYSIS ===\n"

        # Build the final prompt
        lang_instruction = "English" if language == "en" else "the farmer's selected language"
        prompt = (
            f"{system_instructions}\n"
            f"{evidence_section}\n"
            f"{image_section}\n"
            f"Farmer's Question: {farmer_question}\n\n"
            f"Respond in {lang_instruction} "
            f"with a clear, structured answer including:\n"
            f"1. Brief assessment/answer\n"
            f"2. Specific recommendations (with dosages/timing where applicable)\n"
            f"3. Warnings or precautions\n"
            f"4. Verified Sources section (list each source used)\n"
            f"5. Confidence level of your recommendation\n"
        )

        return prompt

    def get_statistics(self) -> Dict:
        """Return knowledge base statistics for the dashboard."""
        return {
            "total_documents": self.kb.total_documents,
            "crops_covered": len(self.kb.get_all_crops()),
            "categories": self.kb.get_all_categories(),
            "crops": self.kb.get_all_crops(),
        }
