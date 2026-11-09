# ---- runtime: the app itself ----
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    AWS_EC2_METADATA_DISABLED=true \
    DATA_DIR=/app/data

# Match your host user so files written to ./data stay owned by you.
ARG UID=1000
ARG GID=1000
RUN groupadd --gid ${GID} app && useradd --uid ${UID} --gid ${GID} --create-home app

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
RUN mkdir -p /app/data && chown app:app /app/data

USER app
CMD ["python", "-m", "app"]

# ---- test: runtime + test tools, run with `docker compose run --rm test` ----
FROM runtime AS test

USER root
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt
COPY pytest.ini .
COPY tests ./tests

USER app
CMD ["pytest"]
