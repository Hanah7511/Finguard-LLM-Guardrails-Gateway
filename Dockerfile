# One image for both doors. docker-compose.yml chooses which door each container runs.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/finguard/.cache/huggingface

WORKDIR /app

# Copied before the code, so changing the code does not reinstall everything.
COPY requirements.txt .
RUN pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY src/ src/
COPY scripts/ scripts/
COPY data/ data/
COPY alembic.ini .
COPY migrations/ migrations/

# Run as an ordinary user: if the app is ever broken into, the attacker is not root.
RUN useradd --create-home --uid 10001 finguard \
    && mkdir -p logs /home/finguard/.cache/huggingface \
    && chown -R finguard:finguard logs /home/finguard/.cache
USER finguard

EXPOSE 8000 8001
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]