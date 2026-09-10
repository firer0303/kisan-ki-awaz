"""
Kisan Ki Awaz - AI Farming Assistant
======================================
Main Streamlit application.
Run with:  streamlit run app.py
"""
import sys
import uuid
from pathlib import Path

# Ensure project root is on path for imports
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from PIL import Image

from services.language_service import LanguageService, SupportedLanguage
from services.recommendation_engine import RecommendationEngine

# ──────────────────────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Kisan Ki Awaz - AI Farming Assistant",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# Load CSS
# ──────────────────────────────────────────────────────────────
css_path = PROJECT_ROOT / "assets" / "css" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# Session State Initialization
# ──────────────────────────────────────────────────────────────
def init_session():
    """Initialize all session state variables."""
    defaults = {
        "language_selected": False,
        "language": None,
        "engine": None,
        "lang_service": None,
        "analysis_result": None,
        "show_explain": False,
        "current_page": "main",
        "demo_running": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session()


# ──────────────────────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────────────────────
def get_engine() -> RecommendationEngine:
    """Get or create the recommendation engine (lazy init)."""
    if st.session_state.engine is None:
        st.session_state.engine = RecommendationEngine()
    return st.session_state.engine


def get_lang_service() -> LanguageService:
    """Get or create the language service."""
    if st.session_state.lang_service is None:
        st.session_state.lang_service = LanguageService()
    return st.session_state.lang_service


def translate(key: str) -> str:
    """Translate a UI label using the language service."""
    ls = get_lang_service()
    return ls.translate(key)


def set_language(lang: SupportedLanguage):
    """Set the session language and initialize services."""
    ls = get_lang_service()
    ls.set_language(lang)
    st.session_state.language_selected = True
    st.session_state.language = lang
    engine = get_engine()
    engine.language_service.set_language(lang)


def reset_session():
    """Reset the entire session."""
    st.session_state.language_selected = False
    st.session_state.language = None
    st.session_state.engine = None
    st.session_state.lang_service = None
    st.session_state.analysis_result = None
    st.session_state.show_explain = False
    st.session_state.current_page = "main"
    st.rerun()


# ──────────────────────────────────────────────────────────────
# Render: Language Selection Screen
# ──────────────────────────────────────────────────────────────
def render_language_selection():
    """Render the language selection screen."""
    st.markdown("""
    <div class="app-header">
        <h1>🌾 Kisan Ki Awaz</h1>
        <p>AI Farming Assistant for Pakistani Farmers</p>
        <p style="font-size:0.95rem; opacity:0.8;">
            کسان کی آواز | هاريءَ جو آواز | کسان دی اواز | د بزگر آواز
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Select Your Language / اپنی زبان منتخب کریں")
    st.markdown("Choose your preferred language. This will be used throughout the entire session.")

    ls = LanguageService()
    languages = ls.get_available_languages()

    cols = st.columns(3)
    lang_list = list(languages.items())

    for i, (lang_enum, lang_config) in enumerate(lang_list):
        with cols[i % 3]:
            native = lang_config.name_native
            english = lang_config.name_en
            btn_label = f"{native}\n{english}"
            if st.button(btn_label, key=f"lang_{lang_enum.value}", use_container_width=True):
                set_language(lang_enum)
                st.rerun()


# ──────────────────────────────────────────────────────────────
# Render: Sidebar
# ──────────────────────────────────────────────────────────────
def render_sidebar():
    """Render the navigation sidebar."""
    with st.sidebar:
        lang_config = get_lang_service().current_config
        st.markdown(f"### 🌾 {translate('app_title')}")
        st.markdown(f"**{translate('app_subtitle')}**")
        st.markdown(f"**Language:** {lang_config.name_native} ({lang_config.name_en})")
        st.divider()

        page = st.radio(
            "Navigation",
            [
                "🏠 Home",
                "🌾 Crop Analysis",
                "🦠 Disease / Pest",
                "📊 Market Intelligence",
                "🌤️ Weather / Risk",
                "📜 History",
                "🎯 Judge Demo",
                "⚙️ System Status",
            ],
            label_visibility="collapsed",
        )
        st.divider()
        if st.button("🔄 Change Language", use_container_width=True):
            reset_session()

        page_map = {
            "🏠 Home": "main",
            "🌾 Crop Analysis": "crop",
            "🦠 Disease / Pest": "disease",
            "📊 Market Intelligence": "market",
            "🌤️ Weather / Risk": "weather",
            "📜 History": "history",
            "🎯 Judge Demo": "demo",
            "⚙️ System Status": "status",
        }
        st.session_state.current_page = page_map.get(page, "main")


# ──────────────────────────────────────────────────────────────
# Render: Main Home Page
# ──────────────────────────────────────────────────────────────
def render_home():
    """Render the main home page with input options."""
    st.markdown(f"## {translate('app_title')}")
    st.markdown(f"**{translate('app_subtitle')}**")
    st.markdown("---")

    st.markdown(f"### {translate('select_language')}: ✅ {get_lang_service().current_config.name_native}")

    st.markdown("---")
    st.markdown("### Choose Your Input Method")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 🎤 Voice / Text Input")
        st.markdown("Ask an agricultural question by typing or voice.")
        question = st.text_area(
            "Your farming question:",
            placeholder="e.g., How to control wheat leaf rust?",
            height=100,
            key="voice_question",
        )
        if st.button("📤 Submit Question", use_container_width=True, type="primary"):
            if question.strip():
                with st.spinner(translate("analyzing")):
                    engine = get_engine()
                    result = engine.process_voice_query(question.strip())
                    st.session_state.analysis_result = result
                    st.rerun()
            else:
                st.warning("Please enter a question first.")

    with col2:
        st.markdown("#### 📷 Take Picture / Upload")
        uploaded_file = st.file_uploader(
            "Upload a crop/plant image:",
            type=["jpg", "jpeg", "png", "webp"],
            key="image_upload",
        )
        optional_q = st.text_input(
            "Optional question about the image:",
            placeholder="e.g., What disease is this?",
            key="image_question",
        )
        if st.button("🔍 Analyze Image", use_container_width=True, type="primary"):
            if uploaded_file:
                with st.spinner(translate("analyzing")):
                    img = Image.open(uploaded_file)
                    engine = get_engine()
                    result = engine.process_image(img, optional_q, "upload")
                    st.session_state.analysis_result = result
                    st.rerun()
            else:
                st.warning("Please upload an image first.")

    with col3:
        st.markdown("#### 📱 Camera Capture")
        st.markdown(
            "Use your device camera to capture a crop image. "
            "Works best on mobile devices."
        )
        camera_image = st.camera_input("Take a photo of your crop:", key="camera_input")
        if camera_image:
            with st.spinner(translate("analyzing")):
                img = Image.open(camera_image)
                engine = get_engine()
                result = engine.process_image(img, "", "camera")
                st.session_state.analysis_result = result
                st.rerun()

    # Display analysis result if available
    if st.session_state.analysis_result:
        st.markdown("---")
        render_analysis_result(st.session_state.analysis_result)


# ──────────────────────────────────────────────────────────────
# Render: Analysis Result
# ──────────────────────────────────────────────────────────────
def render_analysis_result(result: dict):
    """Render a complete analysis result."""
    if not result.get("success"):
        st.error(result.get("error", "Analysis failed."))
        return

    # Demo banner if applicable
    if result.get("is_demo"):
        st.markdown(
            '<div class="demo-banner">🧪 DEMO MODE: Using simulated AI models. '
            "Connect real models/APIs for production use.</div>",
            unsafe_allow_html=True,
        )

    # Image display if image analysis
    vision = result.get("vision_result")
    if vision:
        col_img, col_info = st.columns([1, 2])
        with col_img:
            st.markdown("### 📸 Analyzed Image")
            if st.session_state.get("_uploaded_img"):
                st.image(st.session_state._uploaded_img, use_container_width=True)
        with col_info:
            render_vision_details(vision)
    else:
        st.markdown("### 🎤 Voice/Text Analysis Result")

    # Response text
    lang_config = get_lang_service().current_config
    direction = "rtl" if lang_config.rtl else "ltr"
    st.markdown(
        f'<div style="direction:{direction}; font-family:{lang_config.font_family}; '
        f'padding:16px; background:#f5f5f5; border-radius:10px; line-height:1.8;">'
        f'{result["response_text"]}</div>',
        unsafe_allow_html=True,
    )

    # Verified Sources
    sources = result.get("sources", [])
    if sources:
        st.markdown(f"### ✅ {translate('verified_sources')}")
        for src in sources:
            st.markdown(
                f'<div class="source-card">'
                f'<div class="source-org">🏛️ {src.get("organization", "N/A")}</div>'
                f'<div class="source-doc">📄 {src.get("document_title", "N/A")}</div>'
                f'<div class="source-url">🔗 {src.get("url", "N/A")} '
                f'| Published: {src.get("publication_date", "N/A")} '
                f'| Credibility: {src.get("credibility", "N/A")}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            f'<div class="warning-box">⚠️ {translate("no_source")}</div>',
            unsafe_allow_html=True,
        )

    # Explainable AI button
    st.markdown("---")
    if st.button(f"🔍 {translate('why_this')}", key="explain_btn"):
        st.session_state.show_explain = not st.session_state.show_explain

    if st.session_state.show_explain:
        render_explainability(result)

    # Warnings
    warnings = []
    if vision:
        warnings = vision.get("warnings", [])
    for w in warnings:
        st.markdown(f'<div class="warning-box">⚠️ {w}</div>', unsafe_allow_html=True)

    # Auto-Narration
    narration_html = result.get("narration_html", "")
    if narration_html:
        st.markdown("---")
        st.markdown("### 🔊 Automatic Voice Narration")
        st.markdown(narration_html, unsafe_allow_html=True)

    # Clear button
    st.markdown("---")
    if st.button("🗑️ Clear Result & Start New Analysis", use_container_width=True):
        st.session_state.analysis_result = None
        st.session_state.show_explain = False
        st.rerun()


# ──────────────────────────────────────────────────────────────
# Render: Vision Analysis Details
# ──────────────────────────────────────────────────────────────
def render_vision_details(vision: dict):
    """Render vision model prediction details."""
    st.markdown(f"### {translate('crop_analysis')}")
    st.markdown(f"**Detected:** {vision.get('prediction', 'Unknown')}")

    # Confidence meter
    conf = vision.get("confidence", 0)
    fill_class = "high" if conf >= 70 else ("medium" if conf >= 50 else "low")
    st.markdown(
        f"**{translate('confidence')}:** {conf}%\n"
        f'<div class="confidence-bar">'
        f'<div class="confidence-fill-{fill_class}" '
        f'style="width:{conf}%; height:100%;"></div></div>',
        unsafe_allow_html=True,
    )

    # Risk badge
    risk = vision.get("risk_level", "unknown")
    st.markdown(
        f'**{translate("risk_level")}:** '
        f'<span class="risk-badge risk-{risk}">{risk.upper()}</span>',
        unsafe_allow_html=True,
    )

    st.markdown(f"**Crop:** {vision.get('crop_detected', 'Unknown')}")
    disease = vision.get("disease_detected")
    st.markdown(f"**Disease:** {disease or 'None detected'}")

    # Top predictions
    all_preds = vision.get("all_predictions", [])
    if all_preds and len(all_preds) > 1:
        with st.expander("All Predictions"):
            for label, c in all_preds:
                st.markdown(f"- **{label}**: {c}%")

    # Image quality
    quality = vision.get("image_quality", {})
    if quality:
        with st.expander("Image Quality Assessment"):
            st.markdown(f"- Resolution: {quality.get('resolution', 'N/A')}")
            st.markdown(f"- Brightness: {quality.get('brightness', 'N/A')}")
            st.markdown(f"- Contrast: {quality.get('contrast', 'N/A')}")
            st.markdown(f"- Blurry: {'Yes' if quality.get('is_blurry') else 'No'}")
            st.markdown(f"- Dark: {'Yes' if quality.get('is_dark') else 'No'}")


# ──────────────────────────────────────────────────────────────
# Render: Explainable AI
# ──────────────────────────────────────────────────────────────
def render_explainability(result: dict):
    """Render the 'Why this recommendation?' explainability panel."""
    engine = get_engine()
    explanation = engine.get_explainability(result)

    st.markdown("### 🔍 Why this recommendation?")
    st.markdown(
        f'<div class="info-box">📌 {explanation["ai_reasoning_notice"]}</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="explain-panel">', unsafe_allow_html=True)
    for factor in explanation.get("factors", []):
        st.markdown('<div class="explain-factor">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="explain-category">{factor["category"]}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<span class="explain-source-type">{factor["source_type"]}</span>',
            unsafe_allow_html=True,
        )
        for detail in factor.get("details", []):
            st.markdown(f"  • {detail}")
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# Render: Crop Analysis Page
# ──────────────────────────────────────────────────────────────
def render_crop_analysis():
    """Dedicated crop analysis page."""
    st.markdown(f"## 🌾 {translate('crop_analysis')}")
    st.markdown("Upload a crop, leaf, fruit, or plant image for AI analysis.")

    uploaded = st.file_uploader("Upload crop image:", type=["jpg", "jpeg", "png", "webp"], key="crop_upload")
    question = st.text_input("Optional question:", placeholder="e.g., What is wrong with this crop?", key="crop_q")

    if st.button("🔍 Analyze", use_container_width=True, type="primary"):
        if uploaded:
            with st.spinner(translate("analyzing")):
                img = Image.open(uploaded)
                engine = get_engine()
                result = engine.process_image(img, question, "upload")
                st.session_state.analysis_result = result
                st.rerun()
        else:
            st.warning("Please upload an image.")

    if st.session_state.analysis_result and st.session_state.analysis_result.get("vision_result"):
        st.markdown("---")
        render_analysis_result(st.session_state.analysis_result)


# ──────────────────────────────────────────────────────────────
# Render: Disease/Pest Page
# ──────────────────────────────────────────────────────────────
def render_disease_pest():
    """Dedicated disease/pest analysis page."""
    st.markdown(f"## 🦠 {translate('disease_pest')}")
    st.markdown("Upload an image of a diseased or pest-affected plant.")

    uploaded = st.file_uploader("Upload affected plant image:", type=["jpg", "jpeg", "png", "webp"], key="disease_upload")
    question = st.text_input("Describe the symptoms:", key="disease_q")

    if st.button("🔍 Diagnose", use_container_width=True, type="primary"):
        if uploaded:
            with st.spinner(translate("analyzing")):
                img = Image.open(uploaded)
                engine = get_engine()
                result = engine.process_image(img, question, "upload")
                st.session_state.analysis_result = result
                st.rerun()
        else:
            st.warning("Please upload an image.")

    if st.session_state.analysis_result and st.session_state.analysis_result.get("vision_result"):
        st.markdown("---")
        render_analysis_result(st.session_state.analysis_result)


# ──────────────────────────────────────────────────────────────
# Render: Market Intelligence
# ──────────────────────────────────────────────────────────────
def render_market():
    """Market intelligence page."""
    st.markdown(f"## 📊 {translate('market_info')}")

    engine = get_engine()
    crop = st.selectbox(
        "Select crop:",
        ["Wheat", "Rice", "Cotton", "Sugarcane", "Maize"],
        key="market_crop",
    )
    if st.button("Get Market Info", use_container_width=True):
        prices = engine.market_service.get_crop_prices(crop.lower())

        st.markdown(f"### {crop} Market Prices")

        if prices.get("disclaimer"):
            st.markdown(
                f'<div class="warning-box">{prices["disclaimer"]}</div>',
                unsafe_allow_html=True,
            )

        for p in prices.get("prices", []):
            st.markdown(
                f"- **{p['market']}**: {p['price']} / {p['unit']} "
                f"({p['date']})"
            )

        st.markdown(f"**Trend:** {prices.get('trend', 'unknown').upper()}")
        st.markdown(f"*Source: {prices.get('source', 'N/A')}*")

        if prices.get("is_demo"):
            st.info("💡 Connect a live market data API in .env for real-time prices.")


# ──────────────────────────────────────────────────────────────
# Render: Weather / Risk Alerts
# ──────────────────────────────────────────────────────────────
def render_weather():
    """Weather and risk alerts page."""
    st.markdown(f"## 🌤️ {translate('weather_alerts')}")

    engine = get_engine()
    regions = engine.weather_service.get_available_regions()
    region_keys = list(regions.keys())
    region_names = [f"{k}: {v['name']}" for k, v in regions.items()]

    selected = st.selectbox("Select region:", region_names, key="weather_region")
    region_key = region_keys[region_names.index(selected)]

    weather = engine.weather_service.get_weather(region_key)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Temperature", f"{weather['temperature_c']}°C")
    col2.metric("Humidity", f"{weather['humidity_pct']}%")
    col3.metric("Condition", weather["condition"])
    col4.metric("Wind", f"{weather['wind_kph']} km/h")

    st.markdown(f"**Location:** {weather.get('location', 'N/A')}")
    st.markdown(f"**Source:** {weather.get('source', 'N/A')}")

    st.markdown("### Agricultural Advisory")
    st.info(weather.get("agricultural_advisory", "No advisory available."))

    alerts = weather.get("risk_alerts", [])
    if alerts:
        st.markdown("### ⚠️ Risk Alerts")
        for alert in alerts:
            level = alert.get("level", "info")
            color = {"high": "#f44336", "medium": "#ff9800", "low": "#4caf50"}.get(level, "#2196f3")
            st.markdown(
                f'<div style="background:{color}22; border-left:4px solid {color}; '
                f'padding:10px; margin:6px 0; border-radius:6px;">'
                f'<strong>{level.upper()}:</strong> {alert["message"]}</div>',
                unsafe_allow_html=True,
            )

    if weather.get("is_demo"):
        st.info("💡 Configure weather API in .env for real-time data (OpenWeatherMap or PMD).")


# ──────────────────────────────────────────────────────────────
# Render: History
# ──────────────────────────────────────────────────────────────
def render_history():
    """Farmer analysis history page."""
    st.markdown(f"## 📜 {translate('history')}")

    engine = get_engine()
    history = engine.database_service.get_session_history(engine.session_id)

    if not history:
        st.info("No analysis history yet. Submit a query to get started!")
        return

    for record in history:
        with st.expander(
            f"🔍 {record.get('farmer_question', 'Analysis')[:60]} "
            f"| {record.get('language', '')} | {record.get('created_at', '')}"
        ):
            st.markdown(f"**Input Type:** {record.get('input_type', 'N/A')}")
            st.markdown(f"**Crop:** {record.get('crop_detected', 'N/A')}")
            st.markdown(f"**Disease:** {record.get('disease_detected', 'N/A')}")
            st.markdown(f"**Confidence:** {record.get('confidence', 0)}%")
            st.markdown(f"**Risk:** {record.get('risk_level', 'N/A')}")
            st.markdown("**Response:**")
            st.markdown(record.get("response_text", "N/A")[:500])


# ──────────────────────────────────────────────────────────────
# Render: Judge Demo Mode
# ──────────────────────────────────────────────────────────────
def render_demo():
    """Hackathon judge demo mode with full scenario walkthrough."""
    st.markdown("## 🎯 Judge Demo Mode")
    st.markdown(
        '<div class="demo-banner">Complete demonstration: Language Selection → '
        "Input → AI Analysis → Source Retrieval → Recommendation → Narration</div>",
        unsafe_allow_html=True,
    )

    # Architecture Visualization
    st.markdown("### 🏗️ System Architecture")
    st.markdown(
        '<div class="arch-flow">'
        '<div class="arch-node">🧑‍🌾<br>Farmer Input</div>'
        '<div class="arch-arrow">→</div>'
        '<div class="arch-node">🎤📷<br>Speech/Image<br>Processing</div>'
        '<div class="arch-arrow">→</div>'
        '<div class="arch-node">🤖<br>AI Engine<br>(LLM + CV)</div>'
        '<div class="arch-arrow">→</div>'
        '<div class="arch-node">📚<br>RAG Knowledge<br>Base</div>'
        '<div class="arch-arrow">→</div>'
        '<div class="arch-node">💡<br>Recommendation<br>Engine</div>'
        '<div class="arch-arrow">→</div>'
        '<div class="arch-node">🌐<br>Translation</div>'
        '<div class="arch-arrow">→</div>'
        '<div class="arch-node">📝🔊<br>Text + Voice<br>Narration</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Impact Metrics (clearly labeled as demo)
    st.markdown("### 📈 Impact Dashboard *(Prototype Demo Metrics)*")
    st.markdown(
        '<div class="info-box">⚠️ These metrics are for hackathon demonstration '
        "purposes and do not represent real-world deployment statistics.</div>",
        unsafe_allow_html=True,
    )

    engine = get_engine()
    db_stats = engine.database_service.get_stats()
    rag_stats = engine.rag_service.get_statistics()

    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(
        f'<div class="metric-card"><div class="metric-value">'
        f'{db_stats.get("total_analyses", 0)}</div>'
        f'<div class="metric-label">Analyses Completed</div></div>',
        unsafe_allow_html=True,
    )
    col2.markdown(
        f'<div class="metric-card"><div class="metric-value">6</div>'
        f'<div class="metric-label">Supported Languages</div></div>',
        unsafe_allow_html=True,
    )
    col3.markdown(
        f'<div class="metric-card"><div class="metric-value">'
        f'{rag_stats.get("total_documents", 0)}</div>'
        f'<div class="metric-label">Verified Sources</div></div>',
        unsafe_allow_html=True,
    )
    col4.markdown(
        f'<div class="metric-card"><div class="metric-value">'
        f'{rag_stats.get("crops_covered", 0)}</div>'
        f'<div class="metric-label">Crops Covered</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Demo Scenario
    st.markdown("### 🎬 Run Demo Scenario")
    st.markdown(
        "Click below to run a complete demo scenario: "
        "A farmer in Punjab asks about wheat leaf rust."
    )

    if st.button("▶️ Run Demo Scenario", use_container_width=True, type="primary"):
        with st.spinner("Running complete demo pipeline..."):
            # Set language to Urdu for demo
            set_language(SupportedLanguage.URDU)

            # Process a demo query
            engine = get_engine()
            engine.language_service.set_language(SupportedLanguage.URDU)
            result = engine.process_voice_query(
                "My wheat crop has orange-brown spots on leaves. "
                "What is the disease and how to treat it?"
            )
            st.session_state.analysis_result = result
            st.rerun()

    # Key Features Summary
    st.markdown("---")
    st.markdown("### ✨ Key Features Demonstrated")
    features = [
        "🌍 **6 Languages**: Urdu, Sindhi, Punjabi, Pashto, Balochi, English",
        "🎤 **Voice Input**: Speech-to-text in selected language",
        "📷 **Image Analysis**: CV-based crop/disease detection with confidence scores",
        "📚 **RAG System**: Evidence retrieval from FAO, PARC, NARC, and more",
        "🤖 **LLM Reasoning**: Source-grounded AI recommendations",
        "✅ **Verified Sources**: Clear citations with credibility levels",
        "🔊 **Auto-Narration**: Automatic voice in the selected language",
        "🔍 **Explainable AI**: 'Why this recommendation?' transparency",
        "⚠️ **Confidence Handling**: Low-confidence warnings and expert consultation advice",
        "🔌 **Modular Architecture**: Easy to connect Alibaba Cloud AI, real models, and live APIs",
    ]
    for f in features:
        st.markdown(f"- {f}")


# ──────────────────────────────────────────────────────────────
# Render: System Status
# ──────────────────────────────────────────────────────────────
def render_status():
    """System configuration and status page."""
    st.markdown("## ⚙️ System Status")

    from config import settings

    st.markdown("### Service Providers")
    providers = {
        "LLM": settings.llm.active_provider,
        "Speech STT": settings.speech.active_stt_provider,
        "Speech TTS": settings.speech.active_tts_provider,
        "Vision": settings.vision.active_provider,
        "Weather": settings.weather.active_provider,
        "Market Data": settings.market.active_provider,
        "Translation": settings.translation.active_provider,
    }
    for name, provider in providers.items():
        icon = "✅" if provider not in ("demo", "local") else "🧪"
        st.markdown(f"- {icon} **{name}**: `{provider}`")

    st.markdown("### Knowledge Base")
    engine = get_engine()
    rag_stats = engine.rag_service.get_statistics()
    st.markdown(f"- **Documents:** {rag_stats['total_documents']}")
    st.markdown(f"- **Crops:** {rag_stats['crops_covered']} ({', '.join(rag_stats['crops'][:10])})")
    st.markdown(f"- **Categories:** {', '.join(rag_stats['categories'])}")

    st.markdown("### Database")
    db_stats = engine.database_service.get_stats()
    st.markdown(f"- **Total Analyses:** {db_stats['total_analyses']}")
    st.markdown(f"- **Languages Used:** {', '.join(db_stats['languages_used']) or 'None yet'}")

    st.markdown("### Configuration")
    st.info(
        "Configure API keys in `.env` file (copy from `.env.example`).\n\n"
        "Available integrinations:\n"
        "- **LLM**: OpenAI (GPT-4o) or Alibaba DashScope (Qwen)\n"
        "- **Vision**: Custom model or Alibaba Cloud Vision\n"
        "- **Weather**: OpenWeatherMap or Pakistan Meteorological Department\n"
        "- **Market**: Official agriculture data API\n"
        "- **Translation**: DeepL or LibreTranslate\n"
        "- **Speech**: Google Cloud or Azure Speech Services"
    )


# ──────────────────────────────────────────────────────────────
# Main Application Router
# ──────────────────────────────────────────────────────────────
def main():
    """Main application entry point."""
    if not st.session_state.language_selected:
        render_language_selection()
        return

    render_sidebar()

    page = st.session_state.current_page
    if page == "main":
        render_home()
    elif page == "crop":
        render_crop_analysis()
    elif page == "disease":
        render_disease_pest()
    elif page == "market":
        render_market()
    elif page == "weather":
        render_weather()
    elif page == "history":
        render_history()
    elif page == "demo":
        render_demo()
    elif page == "status":
        render_status()


if __name__ == "__main__":
    main()
