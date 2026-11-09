# Data Analysis Agent web demo (Render free web service, Docker).
FROM python:3.12-slim

# uv, pinned. Keep this in step with the local version (`uv --version`) so it reads uv.lock.
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /usr/local/bin/uv

# Run as an unprivileged user.
RUN useradd --create-home --uid 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/app/.venv/bin:$PATH \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg
WORKDIR /home/user/app

# Dependencies first (cached layer), then the code.
COPY --chown=user pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra web --no-install-project
# Build matplotlib's font cache now, so the first analysis after a deploy is not slow.
RUN python -c "import matplotlib.font_manager"
COPY --chown=user src ./src
RUN uv sync --frozen --no-dev --extra web

# Documentation only: Render routes to the port it sets in PORT; locally the app falls back to 8000.
EXPOSE 8000

# GEMINI_API_KEY (this project's own key, from its own Google project) is set as a secret environment
# variable in the Render dashboard; it is never baked into the image.
# data-analysis-web listens on $PORT (else 8000) with the access log off. Uploads stay in memory;
# the Gemini cache is in memory too (SCOPE continued 2), so nothing is written to disk.
CMD ["data-analysis-web"]
