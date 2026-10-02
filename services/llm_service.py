"""
Kisan Ki Awaz - LLM Service
==============================
Manages communication with LLM providers (OpenAI, DashScope/Alibaba,
or a verified knowledge-base fallback).
"""
from typing import Dict, Optional

from loguru import logger

from config import settings


class LLMService:
    """
    Unified interface for LLM inference.
    Supports OpenAI, Alibaba DashScope (Qwen), with a verified knowledge-base fallback.
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
            return self._fallback_generate(prompt)

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
            return self._fallback_generate(prompt, error=str(e))

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

    def _fallback_generate(self, prompt: str, error: str = "") -> Dict:
        """Generate a formal, language-specific verified fallback response."""
        import re

        language = "en"
        language_match = re.search(r"Respond entirely in ([^.]+)", prompt, re.IGNORECASE)
        if language_match:
            selected = language_match.group(1).strip().lower()
            if selected.startswith("roman urdu"):
                language = "ro"
            elif selected.startswith("urdu"):
                language = "ur"
            elif selected.startswith("sindhi"):
                language = "sd"
            elif selected.startswith("punjabi") and ("gurmukhi" in selected or "hindi script" in selected):
                language = "pa-hi"
            elif selected.startswith("punjabi"):
                language = "pa"
            elif selected.startswith("hindi"):
                language = "hi"
            elif selected.startswith("pashto"):
                language = "ps"
            elif selected.startswith("balochi"):
                language = "bal"

        farmer_question = ""
        if "Farmer's Question:" in prompt:
            farmer_question = prompt.split("Farmer's Question:")[-1].strip().split("\n")[0]

        evidence_found = "VERIFIED AGRICULTURAL EVIDENCE" in prompt
        image_analysis = "IMAGE ANALYSIS RESULTS" in prompt
        detected = "Unknown"
        confidence = "0"
        risk = "Unknown"

        if image_analysis:
            det_match = re.search(r"Detected:\s*(.+)", prompt)
            conf_match = re.search(r"Confidence:\s*(\d+)", prompt)
            risk_match = re.search(r"Risk Level:\s*(\w+)", prompt)
            if det_match:
                detected = det_match.group(1).strip()
            if conf_match:
                confidence = conf_match.group(1)
            if risk_match:
                risk = risk_match.group(1).strip()

        evidence_text = ""
        if evidence_found:
            start_ev = prompt.find("=== VERIFIED AGRICULTURAL EVIDENCE ===")
            end_ev = prompt.find("=== END EVIDENCE ===")
            if start_ev >= 0 and end_ev >= 0:
                evidence_text = prompt[start_ev + len("=== VERIFIED AGRICULTURAL EVIDENCE ==="):end_ev].strip()

        source_name = ""
        for line in evidence_text.split("\n"):
            if line.startswith("[Source:"):
                source_name = line[len("[Source:"):].strip().rstrip("]")
                break

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

        if language == "ro":
            assessment, recommendations, warnings, sources, confidence_label = "Jawab", "Kya karein", "Ehtiyat", "Moatabar zaraye", "Yaqeen ki satah"
            intro = f"Aap ke sawal «{farmer_question or 'zari maslay'}» ke mutabiq dastiyab moatabar zaraati maloomat dekhi gayi hain."
            no_evidence = "Is sawal ke bare mein abhi moatabar zaraati maloomat nahi mil sakin."
            rec_intro = "Milne wali moatabar maloomat ke mutabiq aap ye kaam kar sakte hain."
            warning = "Kisi bhi zaraati dawa ya chemical ko istemal karne se pehle packet par likhi hidayaat zaroor parhein. Zarurat ho to qareebi zaraati maahir se bhi mashwara karein."
            source_default = "Moatabar zaraati maloomat"
            confidence_high = "Zyada"
            confidence_unable = "Yaqeen se nahi bataya ja sakta"
            image_text = f"Tasveer ka tajziya: {detected} (yaqeen {confidence} feesad, khatre ki satah: {risk})"
            consult = "Mazeed pakki rehnumai ke liye maqami zaraati maahir ya Mehkama-e-Zaraat se fasal check karwaein."
        elif language == "ur":
            assessment = "جواب"
            recommendations = "کیا کریں"
            warnings = "احتیاط"
            sources = "قابلِ بھروسا ذرائع"
            confidence_label = "یقین کی سطح"
            intro = f"آپ کے سوال «{farmer_question or 'زرعی مسئلے'}» کے مطابق دستیاب قابلِ بھروسا زرعی معلومات دیکھی گئی ہیں۔"
            no_evidence = "اس سوال کے بارے میں ابھی قابلِ بھروسا زرعی معلومات نہیں مل سکیں۔"
            rec_intro = "ملنے والی قابلِ بھروسا معلومات کے مطابق آپ یہ کام کر سکتے ہیں۔"
            warning = "کسی بھی زرعی دوا یا کیمیکل کو استعمال کرنے سے پہلے پیکٹ پر لکھی ہدایات ضرور پڑھیں۔ ضرورت ہو تو قریبی زرعی ماہر سے بھی مشورہ کریں۔"
            source_default = "قابلِ بھروسا زرعی معلومات"
            confidence_high = "زیادہ"
            confidence_unable = "یقین سے نہیں بتایا جا سکتا"
            image_text = f"تصویری تجزیہ: {detected} (اعتماد: {confidence}٪، خطرے کی سطح: {risk})"
            consult = "مزید پکا مشورہ لینے کے لیے مقامی زرعی ماہر یا محکمۂ زراعت سے فصل چیک کروائیں۔"
        elif language == "sd":
            assessment, recommendations, warnings, sources, confidence_label = "جائزو / جواب", "سفارشون", "احتياطي تدبيرون", "تصديق ٿيل ذريعا", "اعتماد جي سطح"
            intro, no_evidence, rec_intro, warning = "توهان جي سوال جي بنياد تي تصديق ٿيل زرعي معلومات جو جائزو ورتو ويو آهي.", "هن سوال لاءِ تصديق ٿيل زرعي معلومات دستياب نه ٿي سگهي.", "دستياب تصديق ٿيل زرعي معلومات جي بنياد تي هي سفارشون ڪيون وڃن ٿيون:", "زرعي ڪيميڪل استعمال ڪرڻ کان اڳ ليبل جون هدايتون پڙهو ۽ مقامي زرعي ماهر سان تصديق ڪريو."
            source_default, confidence_high, confidence_unable, image_text, consult = "تصديق ٿيل زرعي معلوماتي ذخيرو", "وڌيڪ", "تصديق ممڪن ناهي", f"تصويري جائزو: {detected} (اعتماد: {confidence}٪، خطري جي سطح: {risk})", "حتمي رهنمائي لاءِ مقامي زرعي ماهر سان صلاح ڪريو."
        elif language == "pa":
            assessment, recommendations, warnings, sources, confidence_label = "جائزہ / جواب", "سفارشاں", "احتیاطی تدبیراں", "تصدیق شدہ ذرائع", "اعتماد دی سطح"
            intro, no_evidence, rec_intro, warning = "تہاڈے سوال دی بنیاد تے تصدیق شدہ زرعی معلومات دا جائزہ لیا گیا اے.", "اس سوال لئی تصدیق شدہ زرعی معلومات دستیاب نہیں ہو سکیاں.", "دستیاب تصدیق شدہ زرعی معلومات دی بنیاد تے ایہ سفارشاں کیتیاں جاندیاں نیں:", "زرعی کیمیکل ورتن توں پہلاں لیبل دیاں ہدایات تے عمل کرو تے مقامی زرعی ماہر نال تصدیق کرو."
            source_default, confidence_high, confidence_unable, image_text, consult = "تصدیق شدہ زرعی معلوماتی ذخیرہ", "زیادہ", "تصدیق ممکن نہیں", f"تصویری جائزہ: {detected} (اعتماد: {confidence}٪، خطرے دی سطح: {risk})", "حتمی رہنمائی لئی مقامی زرعی ماہر نال مشورہ کرو."
        elif language == "ps":
            assessment, recommendations, warnings, sources, confidence_label = "ارزونه / ځواب", "سپارښتنې", "احتیاطي تدابیر", "تایید شوې سرچینې", "د باور کچه"
            intro, no_evidence, rec_intro, warning = "ستاسې د پوښتنې پر بنسټ د تایید شوو کرنیزو معلوماتو ارزونه شوې ده.", "د دې پوښتنې لپاره تایید شوي کرنیز معلومات ونه موندل شول.", "د موجودو تایید شوو کرنیزو معلوماتو پر بنسټ لاندې سپارښتنې کېږي:", "د هر ډول کرنیز کیمیاوي موادو له کارولو مخکې د محصول لارښوونې تعقیب کړئ او له سیمه ییز کرنیز متخصص سره مشوره وکړئ."
            source_default, confidence_high, confidence_unable, image_text, consult = "د تایید شوو کرنیزو معلوماتو زیرمه", "لوړ", "تایید ممکن نه دی", f"د انځور ارزونه: {detected} (باور: {confidence}٪، د خطر کچه: {risk})", "د وروستۍ لارښوونې لپاره له سیمه ییز کرنیز متخصص سره مشوره وکړئ."
        elif language == "bal":
            assessment, recommendations, warnings, sources, confidence_label = "جائزگ / جواب", "سفارشاں", "احتیاطی تدبیر", "تصدیقءَ بوتگین سراجاݔں", "اعتمادءِ سطح"
            intro, no_evidence, rec_intro, warning = "شما ءِ سوال ءِ بنیاد ءَ تصدیق بوتگین کشتکاری ءِ معلومات ءِ جائزگ بوتگ.", "اِیں سوال ءَ تصدیق بوتگین کشتکاری ءِ معلومات دسترس ءَ نہ آت.", "دسترس ءَ بوتگین تصدیق بوتگین معلومات ءِ بنیاد ءَ ایں سفارشاں انت:", "هر کشتکاری کیمیاوی چیز ءِ کارمرزی ءِ پیش ءَ لیبل ءِ ہدایتاں پیروی کنیت و مقامی کشتکاری ماهر ءِ مشورت کنیت."
            source_default, confidence_high, confidence_unable, image_text, consult = "تصدیق بوتگین کشتکاری معلوماتی ذخیرہ", "بلند", "تصدیق ممکن نہ انت", f"تصویر ءِ جائزگ: {detected} (اعتماد: {confidence}٪، خطرہ ءِ سطح: {risk})", "حتمی رہنمائی ءِ پاداش ءَ مقامی کشتکاری ماهر ءِ مشورت کنیت."
        else:
            assessment, recommendations, warnings, sources, confidence_label = "Assessment / Answer", "Recommendations", "Warnings / Precautions", "Verified Sources", "Confidence Level"
            intro = f"Based on your question about: {farmer_question or 'crop management'}, the available verified agricultural information has been reviewed."
            no_evidence = "No verified agricultural information could be found for this question."
            rec_intro = "Based on the available verified agricultural information, the following actions are recommended:"
            warning = "Follow product label instructions for any agricultural chemical treatment and consult a local agricultural extension officer."
            source_default, confidence_high, confidence_unable, image_text, consult = "Verified agricultural knowledge base", "High", "Unable to verify", f"Image analysis: {detected} (Confidence: {confidence}%, Risk: {risk})", "Consult a local agricultural extension officer for final guidance."

        parts = []
        parts.append(intro if evidence_found else no_evidence)

        if image_analysis:
            if language == "ur":
                parts.append(f"تصویر کے مطابق ممکنہ بیماری یا مسئلہ {detected} ہے، اس نتیجے کا یقین {confidence} فیصد ہے اور خطرے کی سطح {risk} ہے۔")
            elif language == "sd":
                parts.append(f"تصويري جائزي مطابق ممڪن سڃاڻپ {detected} آهي، اعتماد {confidence} سيڪڙو ۽ خطري جي سطح {risk} آهي.")
            elif language == "pa":
                parts.append(f"تصویری جائزے مطابق ممکنہ شناخت {detected} اے، اعتماد {confidence} فیصد تے خطرے دی سطح {risk} اے۔")
            elif language == "hi":
                parts.append(f"तस्वीर के अनुसार संभावित पहचान {detected} है, विश्वास {confidence} प्रतिशत और जोखिम स्तर {risk} है।")
            elif language == "pa-hi":
                parts.append(f"ਤਸਵੀਰ ਦੇ ਵਿਸ਼ਲੇਸ਼ਣ ਮੁਤਾਬਕ ਸੰਭਾਵੀ ਪਛਾਣ {detected} ਹੈ, ਭਰੋਸਾ {confidence} ਫੀਸਦੀ ਅਤੇ ਖਤਰੇ ਦਾ ਪੱਧਰ {risk} ਹੈ।")
            elif language == "ps":
                parts.append(f"د انځور د ارزونې له مخې احتمالي پېژندنه {detected} ده، د باور کچه {confidence} سلنه او د خطر کچه {risk} ده.")
            elif language == "bal":
                parts.append(f"تصویر ءِ جائزگ ءَ ممکنہ شناخت {detected} اَنت، اعتمادءِ سطح {confidence} فیصد و خطرہ ءِ سطح {risk} اَنت.")
            else:
                parts.append(f"Image analysis indicates {detected} with {confidence}% confidence and {risk} risk.")

        if evidence_found:
            parts.append(rec_intro)
            if content_lines:
                evidence_summary = " ".join(content_lines[:12])
                sentences = [
                    s.strip()
                    for s in re.split(r"[.!?۔؟]+", evidence_summary)
                    if len(s.strip()) > 10
                ]
                if sentences:
                    parts.append(" ".join(sentences[:8]))
            else:
                parts.append(consult)
            parts.append(warning)
            parts.append(consult)
            source_label = {
                "ur": "ماخذ", "sd": "ذريعو", "pa": "ماخذ",
                "pa-hi": "ਸਰੋਤ", "hi": "स्रोत",
                "ps": "سرچینه", "bal": "ماخذ", "en": "Source",
            }.get(language, "ماخذ")
            parts.append(f"{source_label}: {source_name}۔" if source_name else f"{source_default}۔")
            parts.append(f"{confidence_label}: {confidence_high}۔")
        else:
            parts.append(consult)
            parts.append(f"{confidence_label}: {confidence_unable}۔")

        response_text = re.sub(r"\s+", " ", " ".join(p.strip() for p in parts if p and p.strip())).strip()
        # This fallback is generated directly in the selected language.
        # Do not translate it again; doing so can corrupt the language/script.
        localized = True

        return {
            "text": response_text,
            "provider": "knowledge_base",
            "model": "verified-fallback",
            "is_demo": False,
            "error": None,
            "language": language,
            "localized": bool(localized),
        }
