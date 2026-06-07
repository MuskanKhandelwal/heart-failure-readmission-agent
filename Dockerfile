# Combined FastAPI + Streamlit container for the HF discharge-planning agent.
# python:3.11-slim is used deliberately (smaller / more stable than 3.13 for
# the scientific stack and container builds).
FROM python:3.11-slim

# supervisor runs both services; build tools for any wheels that need compiling.
RUN apt-get update \
    && apt-get install -y --no-install-recommends supervisor build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (better layer caching), then the rest of the source.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e .

# Application data + assets (model pickles, BM25 index, chroma db, guidelines, evals).
COPY . /app

# Supervisor program config (base supervisord.conf includes conf.d/*.conf).
COPY docker/supervisord.conf /etc/supervisor/conf.d/app.conf

EXPOSE 8000 8501

CMD ["supervisord", "-n", "-c", "/etc/supervisor/supervisord.conf"]
