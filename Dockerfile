FROM python:3.11-slim

# System dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Install Python dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --user -r requirements.txt

# Copy application code
COPY --chown=user . .

# Port: Render sets PORT=10000 automatically; HF Spaces uses 7860
ENV PORT=7860
EXPOSE $PORT

# Start the server (shell form so $PORT is expanded)
CMD uvicorn app:app --host 0.0.0.0 --port $PORT
