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
            "5. Respond entirely in the farmer's selected language. Do not switch to English unless the farmer selected English or a technical term has no natural translation.\n"
            "6. Use simple, clear, everyday language that Pakistani farmers can understand easily. For Urdu, use modern everyday Pakistani Urdu like people commonly speak and read today. Keep sentences short, natural, friendly, and very easy for farmers. Avoid difficult, literary, bookish, bureaucratic, or overly formal words. Prefer simple words such as 'بتائیں', 'آسان طریقہ', 'بیماری کی نشانیاں', 'بچاؤ', 'علاج', 'احتیاط', 'معتبر معلومات', and 'یقین کی سطح'.\n"
            "7. Include specific, actionable recommendations (dosages, timing, methods).\n"
            "8. Always include a 'Verified Sources' section listing the sources used.\n"
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
        language_names = {
            "ur": "Urdu (Pakistan / اردو)",
            "ro": "Roman Urdu (Pakistan)",
            "sd": "Sindhi (سنڌي)",
            "pa": "Pakistani Punjabi in Shahmukhi script (پنجابی / شاہ مکھی)",
            "ps": "Pashto (پښتو)",
            "bal": "Balochi (بلوچی)",
            "en": "English",
        }
        lang_instruction = language_names.get(language, "the farmer's selected language")
        section_headings = {
            "ur": "جواب، کیا کریں، احتیاط، قابلِ بھروسا ذرائع، یقین کی سطح",
            "ro": "Jawab, Kya karein, Ehtiyat, Moatabar zaraye, Yaqeen ki satah",
            "sd": "جائزو / جواب، سفارشون، احتياطي تدبيرون، تصديق ٿيل ذريعا، اعتماد جي سطح",
            "pa": "جائزہ / جواب، سفارشاں، احتیاطی تدبیراں، تصدیق شدہ ذرائع، اعتماد دی سطح",
            "ps": "ارزونه / ځواب، سپارښتنې، احتیاطي تدابیر، تایید شوې سرچینې، د باور کچه",
            "bal": "جائزگ / جواب، سفارشاں، احتیاطی تدبیر، تصدیقءَ بوتگین سراجاݔں، اعتمادءِ سطح",
            "en": "Assessment / Answer, Recommendations, Warnings / Precautions, Verified Sources, Confidence Level",
        }
        headings = section_headings.get(language, section_headings["en"])
        prompt = (
            f"{system_instructions}\n"
            f"{evidence_section}\n"
            f"{image_section}\n"
            f"Farmer's Question: {farmer_question}\n\n"
            f"Respond entirely in {lang_instruction}. Use the actual selected language and its natural vocabulary and script. "
            f"Every section heading must ALSO be written in the selected language; never use English section headings for Urdu, Sindhi, Punjabi, Pashto, or Balochi. "
            f"Use these section concepts in this order: {headings}. "
            f"For Urdu, Sindhi, Punjabi, Pashto, and Balochi, do not use English or Hindi prose. For Roman Urdu, use natural Pakistani Roman Urdu in Latin letters only; do not use Urdu, Hindi, Devanagari, or Gurmukhi script. Translate or transliterate technical terms into the selected language. "
            f"Include a clear answer, practical recommendations with dosage/timing where applicable, simple precautions, reliable sources, and a simple confidence statement. Return the final farmer answer as one smooth paragraph with no bullets, numbered lists, Markdown headings, or separate list lines. For Urdu, use modern everyday Pakistani Urdu throughout. For Roman Urdu, use natural everyday Pakistani Roman Urdu, not formal transliteration.\n"
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
