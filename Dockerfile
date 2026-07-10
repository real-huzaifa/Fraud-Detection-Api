# Start from a slim Python base — smaller image, faster deploys
FROM python:3.12-slim

# Install system libraries LightGBM needs at runtime (libgomp = OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Install dependencies first (Docker caches this layer if requirements don't change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and the model
COPY src/ ./src/
COPY models/ ./models/

# Expose the port the API runs on
EXPOSE 8000

# Bind to 0.0.0.0 (NOT 127.0.0.1) so the API is reachable from outside the container
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]