FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Create non-root user
RUN addgroup --system app && adduser --system --ingroup app app

# Install dependencies first (cache-friendly)
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Copy project
COPY --chown=app:app . /app

USER app

EXPOSE 8787

# No CMD here (compose will control it)