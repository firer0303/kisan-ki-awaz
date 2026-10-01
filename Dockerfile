# Kisan Ki Awaz - FastAPI Backend Docker Image
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV, audio, and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list first for layer caching
COPY requirements-render.txt .

RUN pip install --no-cache-dir -r requirements-render.txt

# Copy application code
COPY . .

# Railway will provide the PORT environment variable
EXPOSE 8000

# Start the FastAPI application
ENTRYPOINT ["/bin/sh", "-c"]
CMD ["uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
