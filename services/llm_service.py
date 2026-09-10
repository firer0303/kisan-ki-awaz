"""
Kisan Ki Awaz - LLM Service
==============================
Manages communication with LLM providers (OpenAI, DashScope/Alibaba,
or a local demo fallback).
"""
from typing import Dict, Optional

from loguru import logger

from config import settings


class LLMService:
    """
    Unified interface for LLM inference.
    Supports OpenAI, Alibaba DashScope (Qwen), and a local demo mode.
    """

    def __init__(self):
        self.provider = settings.llm.active_provider
        logger.info(f"LLM service initialized (provider: {self.provider})")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1500,
        temperature: float = 0.3,
    ) -> Dict:
        """
        Generate a response from the LLM.

        Returns:
            Dict with keys:
            - text: str
            - provider: str
            - model: str
            - is_demo: bool
            - error: str or None
        """
        if self.provider == "openai":
            return self._openai_generate(prompt, max_tokens, temperature)
        elif self.provider == "dashscope":
            return self._dashscope_generate(prompt, max_tokens, temperature)
        else:
            return self._demo_generate(prompt)

    def _openai_generate(
        self, prompt: str, max_tokens: int, temperature: float
    ) -> Dict:
        """Generate using OpenAI API."""
        try:
            from openai import OpenAI

            client = OpenAI(api_key=settings.llm.openai_api_key)
            response = client.chat.completions.create(
                model=settings.llm.openai_model,
                messages=[
                    {"role": "system", "content": "You are Kisan Ki Awaz, a farming AI assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return {
                "text": response.choices[0].message.content,
                "provider": "openai",
                "model": settings.llm.openai_model,
                "is_demo": False,
                "error": None,
            }
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return self._demo_generate(prompt)

    def _dashscope_generate(
        self, prompt: str, max_tokens: int, temperature: float
    ) -> Dict:
        """Generate using Alibaba Cloud DashScope (Qwen) API."""
        try:
            # Uncomment when dashscope package is installed
            # import dashscope
            # from dashscope import Generation
            # response = Generation.call(
            #     model=settings.llm.dashscope_model,
            #     messages=[{"role": "user", "content": prompt}],
            #     api_key=settings.llm.dashscope_api_key,
            # )
            # return {
            #     "text": response.output.text,
            #     "provider": "dashscope",
            #     "model": settings.llm.dashscope_model,
            #     "is_demo": False,
            #     "error": None,
            # }
            return self._demo_generate(prompt)
        except Exception as e:
            logger.error(f"DashScope API error: {e}")
            return self._demo_generate(prompt)

    def _demo_generate(self, prompt: str) -> Dict:
        """
        Demo LLM fallback. Extracts key information from the prompt
        and generates a structured response based on the retrieved evidence.
        """
        # Parse the prompt to extract information
        farmer_question = ""
        if "Farmer's Question:" in prompt:
            farmer_question = prompt.split("Farmer's Question:")[-1].strip().split("\n")[0]

        evidence_found = "VERIFIED AGRICULTURAL EVIDENCE" in prompt
        image_analysis = "IMAGE ANALYSIS RESULTS" in prompt

        # Extract image analysis details if present
        detected = "Unknown"
        confidence = "0"
        risk = "Unknown"
        if image_analysis:
            import re
            det_match = re.search(r"Detected:\s*(.+)", prompt)
            conf_match = re.search(r"Confidence:\s*(\d+)", prompt)
            risk_match = re.search(r"Risk Level:\s*(\w+)", prompt)
            if det_match:
                detected = det_match.group(1).strip()
            if conf_match:
                confidence = conf_match.group(1)
            if risk_match:
                risk = risk_match.group(1).strip()

        # Extract evidence content
        evidence_text = ""
        if evidence_found:
            start = prompt.find("=== VERIFIED AGRICULTURAL EVIDENCE ===")
            end = prompt.find("=== END EVIDENCE ===")
            if start >= 0 and end >= 0:
                evidence_text = prompt[start + len("=== VERIFIED AGRICULTURAL EVIDENCE ==="):end].strip()

        # Build a structured response
        response_parts = []

        if evidence_found:
            # Extract the first source info
            source_line = ""
            if "Source:" in evidence_text:
                for line in evidence_text.split("\n"):
                    if "Source:" in line:
                        source_line = line.strip()
                        break

            response_parts.append(
                f"## Assessment\n\n"
                f"Based on your question about: *{farmer_question or 'crop management'}*\n\n"
            )

            if image_analysis:
                response_parts.append(
                    f"**Image Analysis:** {detected} "
                    f"(Confidence: {confidence}%, Risk: {risk})\n\n"
                )

            response_parts.append(
                "## Recommendations\n\n"
                "Based on verified agricultural sources, here are the recommended actions:\n\n"
            )

            # Extract key content from evidence
            content_lines = []
            in_content = False
            for line in evidence_text.split("\n"):
                if line.startswith("Content:"):
                    in_content = True
                    content_lines.append(line.replace("Content:", "").strip())
                elif in_content and line.strip() and not line.startswith("---"):
                    content_lines.append(line.strip())
                elif line.startswith("---") and in_content:
                    break

            if content_lines:
                full_content = " ".join(content_lines[:5])
                # Split into bullet points
                sentences = [s.strip() for s in full_content.split(".") if len(s.strip()) > 10]
                for sentence in sentences[:6]:
                    response_parts.append(f"- {sentence.strip()}\n")
                response_parts.append("\n")
            else:
                response_parts.append(
                    "- Follow recommended cultivation practices from local agricultural extension\n"
                    "- Monitor crop health regularly\n"
                    "- Maintain proper irrigation and fertilization schedules\n\n"
                )

            response_parts.append(
                "## Warnings\n\n"
                "- Always follow label instructions for any chemical treatments\n"
                "- Consult your local agricultural extension officer for site-specific advice\n"
            )
            if image_analysis and int(confidence) < 50:
                response_parts.append(
                    "- ⚠️ LOW CONFIDENCE: This image analysis is preliminary. "
                    "Please get expert confirmation before taking action.\n"
                )
            response_parts.append("\n")

            # Verified Sources section
            response_parts.append("## Verified Sources\n\n")
            sources = []
            for line in evidence_text.split("\n"):
                if line.startswith("[Source:"):
                    src = line.strip("[]")
                    sources.append(f"- {src}\n")
            if sources:
                response_parts.extend(sources[:3])
            else:
                response_parts.append("- Agricultural knowledge base (verified sources)\n")

            response_parts.append(f"\n**Confidence Level:** {'High' if evidence_found else 'Medium'}\n")
        else:
            response_parts.append(
                "## Assessment\n\n"
                "I could not find verified information in the agricultural knowledge base "
                f"for your question: *{farmer_question or 'your query'}*\n\n"
                "## Recommendation\n\n"
                "Please consult your local agricultural extension officer or "
                "visit the nearest Agriculture Department office for expert advice.\n\n"
                "**Confidence Level:** Unable to verify\n"
            )

        response_text = "".join(response_parts)

        return {
            "text": response_text,
            "provider": "demo",
            "model": "local-demo",
            "is_demo": True,
            "error": None,
        }
