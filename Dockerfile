FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN useradd --create-home --uid 10001 appuser

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./
RUN mkdir -p /app/.data && chown -R appuser:appuser /app

USER appuser

EXPOSE 9004

CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9004"]
