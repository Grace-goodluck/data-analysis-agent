"""FastAPI web demo over the unchanged analysis (decisions 4–5; continued 1–4).

Run with `data-analysis-web` (main() below): listens on $PORT, which Render sets, else 8000.

create_app() takes every dependency as an argument so tests pass fakes;
build_default_app() wires the real Gemini client from environment variables.

Privacy (decision 5): uploads are analysed in memory and never stored. The multipart parser
is told to keep uploads in memory instead of spooling them to a temporary file, and the size
is checked from Content-Length before the body is read. Logs carry only the time, row and
column counts and a result label — never the file name, column names or any value.
"""

import logging
import os
import threading
from dataclasses import dataclass, field, replace

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartParser

from data_analysis_agent.analyse import analyse_dataframe
from data_analysis_agent.config import MAX_ROWS, MAX_UPLOAD_BYTES, WEB_GEMINI_RETRY_WAITS_S, load_config
from data_analysis_agent.gemini_client import GeminiLLM, MemoryCache, make_gemini_call
from data_analysis_agent.load import LoadError, load_csv, size_label
from data_analysis_agent.models import Report
from data_analysis_agent.pacing import Pacer
from data_analysis_agent.ports import LLM
from data_analysis_agent.report import ENV, render_html, report_filename
from data_analysis_agent.web.jobs import Executor, JobStore
from data_analysis_agent.web.limits import Limiter, Limits

log = logging.getLogger("data_analysis_agent.web")
TEMPLATES = Jinja2Templates(env=ENV)  # the report's environment: autoescape on, same filters

FORM_OVERHEAD_BYTES = 64 * 1024  # multipart boundaries and headers around the file

PUBLIC_REASONS = {
    "no_key": "the AI summary is not configured on this demo",
    "rejected": "the AI summary contained a number that is not in the computed statistics, so it was not shown",
}
PUBLIC_BUSY = "the AI service is busy or failed"
UNREADABLE_UPLOAD = "The upload could not be read. Please try again."


@dataclass(frozen=True)
class WebSettings:
    gemini_retry_waits_s: tuple[float, ...] = WEB_GEMINI_RETRY_WAITS_S  # continued 1: one retry after 10 s
    limits: Limits = field(default_factory=Limits)
    max_upload_bytes: int = MAX_UPLOAD_BYTES
    max_concurrent_analyses: int = 2


class SerialLLM:
    """Gemini calls one at a time: analyses run on worker threads and the Pacer is not thread-safe."""

    def __init__(self, inner: LLM) -> None:
        self.model = inner.model
        self._inner = inner
        self._lock = threading.Lock()

    def generate(self, prompt: str) -> str:
        with self._lock:
            return self._inner.generate(prompt)


def visitor_id(request: Request) -> str:
    """First IP in X-Forwarded-For (Render's proxy), else the direct peer."""
    first = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    return first or (request.client.host if request.client else "unknown")


def public_report(report: Report) -> Report:
    """Replace internal failure detail (Gemini errors, the rejected number) with a public reason."""
    if report.summary.available:
        return report
    reason = PUBLIC_REASONS.get(report.summary.code, PUBLIC_BUSY)
    return replace(report, summary=replace(report.summary, reason=reason))


def result_label(report: Report) -> str:
    summary = "summary" if report.summary.available else f"no_summary:{report.summary.code or 'unknown'}"
    return f"rows={report.stats.rows} cols={report.stats.columns} charts={len(report.charts)} {summary}"


def create_app(
    *,
    llm: LLM | None,
    limiter: Limiter,
    settings: WebSettings = WebSettings(),
    executor: Executor | None = None,
) -> FastAPI:
    # Keep whole uploads in memory (never a temporary file on disk); Content-Length is checked first.
    MultiPartParser.spool_max_size = settings.max_upload_bytes + FORM_OVERHEAD_BYTES

    app = FastAPI(title="Data Analysis Agent", docs_url=None, redoc_url=None, openapi_url=None)
    serial_llm = SerialLLM(llm) if llm is not None else None

    def run(df: pd.DataFrame, source_name: str) -> Report:
        report = analyse_dataframe(df, source_name, serial_llm)
        log.info("analysis %s", result_label(report))
        return public_report(report)

    jobs = JobStore(run, executor, max_workers=settings.max_concurrent_analyses)

    def page(request: Request, name: str, status: int = 200, headers: dict | None = None, **context) -> Response:
        context = {
            "limits": settings.limits,
            "max_size": size_label(settings.max_upload_bytes),
            "max_rows": MAX_ROWS,
            **context,
        }
        return TEMPLATES.TemplateResponse(request, name, context, status_code=status, headers=headers)

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> Response:
        return page(request, "index.html")

    @app.post("/analyse")
    async def start_analysis(request: Request) -> Response:
        length = request.headers.get("content-length", "")
        if not length.isdigit():
            return page(request, "index.html", 411, error=UNREADABLE_UPLOAD)
        if int(length) > settings.max_upload_bytes + FORM_OVERHEAD_BYTES:
            log.info("rejected result=too_large")
            return page(request, "index.html", 413,
                        error=f"The file is larger than {size_label(settings.max_upload_bytes)}.")

        try:
            form = await request.form(max_files=1, max_fields=5)
        except Exception:  # malformed multipart body
            log.info("rejected result=bad_form")
            return page(request, "index.html", 400, error=UNREADABLE_UPLOAD)
        try:
            upload = form.get("file")
            if not isinstance(upload, UploadFile) or not upload.filename:
                return page(request, "index.html", 400, error="Please choose a CSV file.")
            filename = upload.filename
            data = await upload.read(settings.max_upload_bytes + 1)
        finally:
            await form.close()

        try:  # before the limiter, so a bad file does not use up the visitor's quota
            df = await run_in_threadpool(load_csv, data, filename, settings.max_upload_bytes)
        except LoadError as exc:
            log.info("rejected result=bad_file")
            return page(request, "index.html", 400, error=str(exc))
        finally:
            del data

        refusal = limiter.try_start(visitor_id(request))
        if refusal is not None:
            log.info("refused result=limit")
            return page(request, "index.html", 429, {"Retry-After": str(refusal.retry_after_s)},
                        error=refusal.message)

        job = jobs.submit(df, filename)
        return RedirectResponse(f"/analyse/{job.id}", status_code=303)

    @app.get("/analyse/{job_id}", response_class=HTMLResponse)
    def show_analysis(request: Request, job_id: str) -> Response:
        job = jobs.get(job_id)
        if job is None:
            return page(request, "index.html", 404,
                        error="That analysis was not found. Results are kept for a limited time.")
        return page(request, "result.html", job=job, report=job.report,
                    download_url=f"/analyse/{job.id}/report.html")

    @app.get("/analyse/{job_id}/report.html")
    def download_report(job_id: str) -> Response:
        job = jobs.get(job_id)
        if job is None or job.report is None:
            return PlainTextResponse("Not found", status_code=404)
        return HTMLResponse(
            render_html(job.report),
            headers={"Content-Disposition": f'attachment; filename="{report_filename(job.report)}"'},
        )

    @app.api_route("/health", methods=["GET", "HEAD"])  # UptimeRobot's HTTP monitors send HEAD
    def health() -> Response:
        """For Render and UptimeRobot: answers at once, no analysis, no Gemini call."""
        return PlainTextResponse("ok")

    return app


def build_default_app(executor: Executor | None = None) -> FastAPI:
    """Real wiring. GEMINI_API_KEY (this project's own key) is a Render secret (decision 5)."""
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    log.setLevel(logging.INFO)  # our lines carry only counts and a result label

    settings = WebSettings()
    config = load_config()
    llm = None
    if config.gemini_api_key:
        llm = GeminiLLM(
            config.model,
            make_gemini_call(config.model, config.gemini_api_key),
            MemoryCache(),  # continued 2: nothing written to disk
            Pacer(),
            retry_waits=settings.gemini_retry_waits_s,
        )
    return create_app(llm=llm, limiter=Limiter(settings.limits), settings=settings, executor=executor)


DEFAULT_PORT = 8000


def port_from_env(env: dict[str, str] | None = None) -> int:
    """Render provides the port in PORT; locally fall back to 8000. A bad value fails loudly."""
    value = (os.environ if env is None else env).get("PORT", "").strip()
    return int(value) if value else DEFAULT_PORT


def main() -> None:
    """Entry point `data-analysis-web` (used by the Dockerfile)."""
    import uvicorn  # only in the `web` extra

    uvicorn.run(
        "data_analysis_agent.web.app:build_default_app",
        factory=True,
        host="0.0.0.0",
        port=port_from_env(),
        access_log=False,  # no client IPs or paths in the logs
    )
