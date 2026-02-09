FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

COPY pyproject.toml README.md ./
COPY app ./app
COPY viewer ./viewer
COPY data ./data

ARG INSTALL_MODEL_DEPS=false
RUN pip install --no-cache-dir --upgrade pip \
    && if [ "$INSTALL_MODEL_DEPS" = "true" ]; then \
      pip install --no-cache-dir -e ".[model]"; \
    else \
      pip install --no-cache-dir -e .; \
    fi

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
