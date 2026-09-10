"""
Kisan Ki Awaz - Image Processing Service
==========================================
Handles image upload, camera capture, validation, and preprocessing.
"""
import os
import uuid
from pathlib import Path
from typing import Optional, Tuple

from loguru import logger
from PIL import Image, ImageOps


class ImageService:
    """Handles image input from file uploads and camera capture."""

    MAX_SIZE_MB = 10
    SUPPORTED_FORMATS = {"jpg", "jpeg", "png", "webp", "bmp"}
    TARGET_SIZE = (800, 800)  # Resize large images to this max dimension

    def __init__(self, upload_dir: str = "uploads"):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def validate_uploaded_file(self, file) -> Tuple[bool, str]:
        """Validate an uploaded file before processing."""
        if file is None:
            return False, "No file uploaded."

        # Check file size
        file.seek(0, 2)
        size_mb = file.tell() / (1024 * 1024)
        file.seek(0)

        if size_mb > self.MAX_SIZE_MB:
            return False, f"File too large ({size_mb:.1f}MB). Maximum: {self.MAX_SIZE_MB}MB."

        # Check extension
        name = getattr(file, "name", "unknown")
        ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        if ext not in self.SUPPORTED_FORMATS:
            return False, f"Unsupported format '.{ext}'. Use: {', '.join(self.SUPPORTED_FORMATS)}"

        return True, "Valid"

    def load_image(self, source) -> Optional[Image.Image]:
        """
        Load an image from various sources:
        - Streamlit UploadedFile
        - PIL Image
        - File path string
        """
        try:
            if isinstance(source, Image.Image):
                return source

            if isinstance(source, str):
                return Image.open(source)

            # Streamlit UploadedFile or file-like object
            if hasattr(source, "read"):
                source.seek(0)
                return Image.open(source)

            return None
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            return None

    def preprocess(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for analysis:
        - Auto-orient based on EXIF
        - Resize if too large
        - Convert to RGB
        """
        # Auto-orient
        image = ImageOps.exif_transpose(image)

        # Resize if needed (maintain aspect ratio)
        if image.size[0] > self.TARGET_SIZE[0] or image.size[1] > self.TARGET_SIZE[1]:
            image.thumbnail(self.TARGET_SIZE, Image.Resampling.LANCZOS)

        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    def save_image(self, image: Image.Image, prefix: str = "img") -> str:
        """Save image to uploads directory and return the path."""
        filename = f"{prefix}_{uuid.uuid4().hex[:8]}.jpg"
        filepath = self.upload_dir / filename
        image.save(filepath, "JPEG", quality=85)
        logger.info(f"Image saved: {filepath}")
        return str(filepath)
