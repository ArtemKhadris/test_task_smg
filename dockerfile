# Dockerfile (multi-stage)
# Stage 1: build wheels to make final image smaller and deterministic
FROM python:3.10-slim AS builder

WORKDIR /app

# system deps for building wheels if needed (kept only for builder)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# copy requirements and build wheels
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip wheel --wheel-dir /wheels -r /app/requirements.txt

# Stage 2: runtime image
FROM python:3.10-slim

# create non-root user
RUN useradd --create-home appuser
WORKDIR /app

# copy wheels from builder and install
COPY --from=builder /wheels /wheels
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
 && pip install --no-index --find-links=/wheels -r /app/requirements.txt \
 && rm -rf /wheels /root/.cache/pip

# copy project sources
COPY . /app

# set permissions
RUN chown -R appuser:appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1

# expose port for FastAPI
EXPOSE 8000

# default command
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
