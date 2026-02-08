FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Upgrade pip and make installs more resilient on Render
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
      --default-timeout=200 \
      --retries 10 \
      -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
