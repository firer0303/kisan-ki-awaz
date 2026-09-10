"""
Kisan Ki Awaz - Configuration Management
==========================================
Centralized configuration loaded from environment variables.
Never hard-code API keys or secrets in source code.
"""
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load .env file from project root
PROJECT_ROOT = Path(__file__).parent.resolve()
load_dotenv(PROJECT_ROOT / ".env")


class AppConfig(BaseModel):
    """Application-level settings."""
    env: str = Field(default="development", alias="APP_ENV")
    debug: bool = Field(default=True, alias="APP_DEBUG")
    port: int = Field(default=8501, alias="APP_PORT")
    host: str = Field(default="localhost", alias="APP_HOST")


class LLMConfig(BaseModel):
    """LLM provider configuration."""
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o", alias="OPENAI_MODEL")
    dashscope_api_key: Optional[str] = Field(default=None, alias="DASHSCOPE_API_KEY")
    dashscope_model: str = Field(default="qwen-max", alias="DASHSCOPE_MODEL")

    @property
    def active_provider(self) -> str:
        """Return the first available LLM provider name."""
        if self.openai_api_key:
            return "openai"
        if self.dashscope_api_key:
            return "dashscope"
        return "demo"  # Fallback to demo mode


class SpeechConfig(BaseModel):
    """Speech-to-text and text-to-speech configuration."""
    google_api_key: Optional[str] = Field(default=None, alias="GOOGLE_API_KEY")
    google_tts_lang: str = Field(default="ur-PK", alias="GOOGLE_TTS_LANGUAGE_CODE")
    azure_speech_key: Optional[str] = Field(default=None, alias="AZURE_SPEECH_KEY")
    azure_speech_region: str = Field(default="eastus", alias="AZURE_SPEECH_REGION")

    @property
    def active_stt_provider(self) -> str:
        if self.google_api_key:
            return "google"
        return "local"

    @property
    def active_tts_provider(self) -> str:
        if self.google_api_key:
            return "google"
        return "gtts"


class VisionConfig(BaseModel):
    """Computer vision model configuration."""
    huggingface_token: Optional[str] = Field(default=None, alias="HUGGINGFACE_TOKEN")
    custom_model_path: Optional[str] = Field(default=None, alias="CUSTOM_VISION_MODEL_PATH")
    alibaba_vision_key: Optional[str] = Field(default=None, alias="ALIBABA_VISION_API_KEY")
    default_model: str = "google/vit-base-patch16-224"  # Pretrained agricultural CV model

    @property
    def active_provider(self) -> str:
        if self.custom_model_path:
            return "custom"
        if self.alibaba_vision_key:
            return "alibaba"
        return "pretrained"


class WeatherConfig(BaseModel):
    """Weather data API configuration."""
    pmd_api_key: Optional[str] = Field(default=None, alias="PMD_API_KEY")
    openweathermap_key: Optional[str] = Field(default=None, alias="OPENWEATHERMAP_API_KEY")

    @property
    def active_provider(self) -> str:
        if self.pmd_api_key:
            return "pmd"
        if self.openweathermap_key:
            return "openweathermap"
        return "demo"


class MarketConfig(BaseModel):
    """Market data API configuration."""
    api_key: Optional[str] = Field(default=None, alias="MARKET_DATA_API_KEY")
    api_url: Optional[str] = Field(default=None, alias="MARKET_DATA_API_URL")

    @property
    def active_provider(self) -> str:
        if self.api_key and self.api_url:
            return "api"
        return "demo"


class TranslationConfig(BaseModel):
    """Translation service configuration."""
    deepl_api_key: Optional[str] = Field(default=None, alias="DEEPL_API_KEY")
    libretranslate_url: Optional[str] = Field(default=None, alias="LIBRETRANSLATE_URL")

    @property
    def active_provider(self) -> str:
        if self.deepl_api_key:
            return "deepl"
        if self.libretranslate_url:
            return "libretranslate"
        return "googletrans"


class RAGConfig(BaseModel):
    """RAG knowledge base configuration."""
    knowledge_base_dir: str = Field(default="data/knowledge_base", alias="KNOWLEDGE_BASE_DIR")
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="RAG_EMBEDDING_MODEL"
    )


class DatabaseConfig(BaseModel):
    """Database configuration."""
    url: str = Field(default="sqlite:///kisan_ki_awaz.db", alias="DATABASE_URL")


class Settings(BaseModel):
    """Root settings container for the entire application."""
    app: AppConfig
    llm: LLMConfig
    speech: SpeechConfig
    vision: VisionConfig
    weather: WeatherConfig
    market: MarketConfig
    translation: TranslationConfig
    rag: RAGConfig
    database: DatabaseConfig

    @classmethod
    def from_env(cls) -> "Settings":
        """Build Settings from environment variables."""
        return cls(
            app=AppConfig(
                env=os.getenv("APP_ENV", "development"),
                debug=os.getenv("APP_DEBUG", "true").lower() == "true",
                port=int(os.getenv("APP_PORT", "8501")),
                host=os.getenv("APP_HOST", "localhost"),
            ),
            llm=LLMConfig(
                openai_api_key=os.getenv("OPENAI_API_KEY") or None,
                openai_model=os.getenv("OPENAI_MODEL", "gpt-4o"),
                dashscope_api_key=os.getenv("DASHSCOPE_API_KEY") or None,
                dashscope_model=os.getenv("DASHSCOPE_MODEL", "qwen-max"),
            ),
            speech=SpeechConfig(
                google_api_key=os.getenv("GOOGLE_API_KEY") or None,
                google_tts_lang=os.getenv("GOOGLE_TTS_LANGUAGE_CODE", "ur-PK"),
                azure_speech_key=os.getenv("AZURE_SPEECH_KEY") or None,
                azure_speech_region=os.getenv("AZURE_SPEECH_REGION", "eastus"),
            ),
            vision=VisionConfig(
                huggingface_token=os.getenv("HUGGINGFACE_TOKEN") or None,
                custom_model_path=os.getenv("CUSTOM_VISION_MODEL_PATH") or None,
                alibaba_vision_key=os.getenv("ALIBABA_VISION_API_KEY") or None,
            ),
            weather=WeatherConfig(
                pmd_api_key=os.getenv("PMD_API_KEY") or None,
                openweathermap_key=os.getenv("OPENWEATHERMAP_API_KEY") or None,
            ),
            market=MarketConfig(
                api_key=os.getenv("MARKET_DATA_API_KEY") or None,
                api_url=os.getenv("MARKET_DATA_API_URL") or None,
            ),
            translation=TranslationConfig(
                deepl_api_key=os.getenv("DEEPL_API_KEY") or None,
                libretranslate_url=os.getenv("LIBRETRANSLATE_URL") or None,
            ),
            rag=RAGConfig(
                knowledge_base_dir=os.getenv("KNOWLEDGE_BASE_DIR", "data/knowledge_base"),
                embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            ),
            database=DatabaseConfig(
                url=os.getenv("DATABASE_URL", "sqlite:///kisan_ki_awaz.db"),
            ),
        )


# Singleton settings instance
settings = Settings.from_env()
