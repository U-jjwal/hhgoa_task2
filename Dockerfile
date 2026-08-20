FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (massive size & RAM reduction)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Limit PyTorch to a single thread to save memory and CPU on virtualized hosts
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the sentence-transformers model during build phase (fixes startup timeouts)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

# Copy application code (includes pre-built FAISS index in data/)
COPY . .

# Expose port
EXPOSE 8000

# Production start — no --reload to prevent crash loops
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--log-level", "info"]
