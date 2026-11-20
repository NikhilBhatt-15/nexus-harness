# nexus-agent web app (UI + API). Build from the repo root:
#   docker build -t nexus-agent .
#   docker run -p 8787:8787 -e GEMINI_API_KEYS=k1,k2 -e NXA_SERVER_TOKEN=change-me -v nxa-data:/data nexus-agent
FROM python:3.12-slim

# libgomp1: LightGBM runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY nexus_lite ./nexus_lite
COPY nexus_agent ./nexus_agent
RUN pip install --no-cache-dir ./nexus_lite "./nexus_agent[mcp,sql]" scikit-learn lightgbm openpyxl pyarrow psycopg2-binary pymysql

# NXA_HOME holds projects, uploads, sessions and knowledge bases: mount a volume there to keep them.
# No Ollama in the container, so knowledge bases use BM25 (set NXA_EMBEDDER=litellm:<embedding model> to use an API).
ENV NXA_HOME=/data \
    NXA_EMBEDDER=none \
    PORT=8787 \
    PYTHONUNBUFFERED=1
VOLUME /data
EXPOSE 8787

# A public server must have NXA_SERVER_TOKEN set; the UI asks for it once and remembers it.
CMD ["sh", "-c", "nxa serve --host 0.0.0.0 --port ${PORT}"]
