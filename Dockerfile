# To run locally...
# docker build -t logi-demo .
# docker run --rm -p 8080:8080 logi-demo
# 
# if any environmental variables needed...
# a) include below default key=value
# b) docker run --rm -p 8080:8080 -e key=value logi-demo

# host port eg: docker run --rm -p 8080:8080 -e "OPENAI_BASE_URL=http://host.docker.internal:1234/v1" logi-demo
# env file: docker run --rm -p 8080:8080 --env-file .env logi-demo

# docker build --platform linux/amd64,linux/arm64 -t logi-demo .

FROM python:3.13-slim

# Print logs immediately (Cloud Run collects stdout/stderr)
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install deps first so this layer is cached when only code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# specify any ENV NAME=value

# ONE worker: session state lives in process memory, and each worker
# process would have its own separate copy. Threads handle concurrency.
# Cloud Run sets $PORT; default to 8080 for local runs.
CMD exec gunicorn --bind ":${PORT:-8080}" --workers 1 --threads 8 --timeout 0 logistics_agent.web:app
