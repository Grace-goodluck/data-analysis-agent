"""In-memory analysis jobs (same design as website_audit_agent).

An analysis is quick, but the Gemini call (with a retry) can take tens of seconds,
so the form starts a job and the results page reloads until it is done.
Job ids are random and unguessable: only the person who uploaded has the link.
A job keeps only the finished report — never the uploaded data, which is dropped
as soon as the analysis ends (decision 5).
"""

import logging
import secrets
import threading
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal, Protocol

from data_analysis_agent.models import Report

log = logging.getLogger("data_analysis_agent.web")

JobStatus = Literal["queued", "running", "done", "error"]


@dataclass
class Job:
    id: str
    status: JobStatus = "queued"
    report: Report | None = None


class Executor(Protocol):
    def submit(self, fn: Callable, /, *args): ...


class JobStore:
    def __init__(
        self,
        run: Callable[..., Report],
        executor: Executor | None = None,
        max_workers: int = 2,
        max_jobs: int = 200,
    ) -> None:
        self._run = run
        self._executor = executor or ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="analysis")
        self._max_jobs = max_jobs
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._lock = threading.Lock()

    def submit(self, *args) -> Job:
        job = Job(id=secrets.token_urlsafe(16))
        with self._lock:
            self._jobs[job.id] = job
            while len(self._jobs) > self._max_jobs:  # forget the oldest
                self._jobs.popitem(last=False)
        self._executor.submit(self._work, job, *args)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def _work(self, job: Job, *args) -> None:
        job.status = "running"
        try:
            job.report = self._run(*args)
            job.status = "done"
        except Exception as exc:  # the analysis should not raise; never log the message (it may hold data)
            log.error("analysis result=error:%s", type(exc).__name__)
            job.status = "error"
