"""Exactly one in-process worker backed by the PostgreSQL processing queue."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import func, select

from app.core.config import Settings, get_settings
from app.core.enums import ProcessingStatus
from app.core.logging import get_logger
from app.db.models import ProcessingRun
from app.db.session import SessionLocal
from app.services.job_service import execute_processing_run


logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ClaimedJob:
    run_id: uuid.UUID
    request_id: str | None


@dataclass(frozen=True, slots=True)
class RecoveryReport:
    stale_failed_run_ids: tuple[uuid.UUID, ...]
    queued_run_count: int


@dataclass(frozen=True, slots=True)
class WorkerStatus:
    enabled: bool
    running: bool
    accepting_claims: bool
    active_run_id: uuid.UUID | None
    pending_count: int
    worker_instance_id: str


def _background_metadata(run: ProcessingRun) -> dict[str, Any]:
    raw = (run.configuration_json or {}).get("background_processing")
    return dict(raw) if isinstance(raw, dict) else {}


def _persist_background(run: ProcessingRun, background: dict[str, Any]) -> None:
    configuration = dict(run.configuration_json or {})
    configuration["background_processing"] = background
    run.configuration_json = configuration


def recover_stale_processing_runs(
    settings: Settings | None = None,
    *,
    now: datetime | None = None,
) -> RecoveryReport:
    """Fail only stale running rows from older processes and preserve queued work."""

    configured = settings or get_settings()
    recovered_at = now or datetime.now(timezone.utc)
    cutoff = recovered_at - timedelta(minutes=configured.processing_stale_after_minutes)
    failed: list[uuid.UUID] = []
    with SessionLocal.begin() as session:
        running = list(
            session.scalars(
                select(ProcessingRun)
                .where(ProcessingRun.status == ProcessingStatus.RUNNING.value)
                .with_for_update(skip_locked=True)
            )
        )
        for run in running:
            background = _background_metadata(run)
            claimed_at: datetime | None = None
            raw_claimed_at = background.get("claimed_at")
            if isinstance(raw_claimed_at, str):
                try:
                    claimed_at = datetime.fromisoformat(raw_claimed_at)
                except ValueError:
                    claimed_at = None
            reference = claimed_at or run.started_at
            if reference is not None and reference > cutoff:
                continue
            prior_stage = run.stage
            prior_progress = run.progress_percent
            run.status = ProcessingStatus.FAILED.value
            run.stage = "failed"
            run.completed_at = recovered_at
            if run.started_at is not None:
                run.duration_ms = max(
                    0, round((recovered_at - run.started_at).total_seconds() * 1000)
                )
            run.error_message = "Processing was interrupted by a prior application process."
            background["recovery"] = {
                "category": "process_interrupted",
                "recovered_at": recovered_at.isoformat(),
                "prior_status": ProcessingStatus.RUNNING.value,
                "prior_stage": prior_stage,
                "prior_progress_percent": prior_progress,
            }
            _persist_background(run, background)
            failed.append(run.id)
        queued_count = int(
            session.scalar(
                select(func.count())
                .select_from(ProcessingRun)
                .where(ProcessingRun.status == ProcessingStatus.QUEUED.value)
            )
            or 0
        )
    if failed:
        logger.warning("Recovered %d stale processing run(s)", len(failed))
    return RecoveryReport(tuple(failed), queued_count)


class ProcessingWorker:
    """Poll, atomically claim, and execute one PostgreSQL-backed run at a time."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._worker_instance_id = str(uuid.uuid4())
        self._state_lock = threading.Lock()
        self._wake_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._active_run_id: uuid.UUID | None = None
        self._accepting_claims = False

    def start(self, *, recover_persisted_jobs: bool = False) -> None:
        """Start the single polling thread; recovery is explicit and idempotent."""

        if not self._settings.processing_worker_enabled:
            logger.info("Processing worker is disabled by configuration")
            return
        if recover_persisted_jobs:
            recover_stale_processing_runs(self._settings)
        with self._state_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._wake_event.clear()
            self._accepting_claims = True
            self._thread = threading.Thread(
                target=self._run,
                name="parse-before-you-prompt-worker",
                daemon=True,
            )
            self._thread.start()
        logger.info("Started one processing worker instance_id=%s", self._worker_instance_id)

    def enqueue(self, run_id: uuid.UUID, request_id: str | None = None) -> bool:
        """Wake the poller; PostgreSQL remains the authoritative queue."""

        if not self._settings.processing_worker_enabled:
            return False
        logger.info("request_id=%s queued_run_id=%s", request_id or "unavailable", run_id)
        self._wake_event.set()
        return True

    def status(self) -> WorkerStatus:
        with self._state_lock:
            thread = self._thread
            active = self._active_run_id
            accepting = self._accepting_claims
        pending = 0
        try:
            with SessionLocal() as session:
                pending = int(
                    session.scalar(
                        select(func.count())
                        .select_from(ProcessingRun)
                        .where(ProcessingRun.status == ProcessingStatus.QUEUED.value)
                    )
                    or 0
                )
        except Exception:
            logger.warning("Unable to count queued processing runs")
        return WorkerStatus(
            enabled=self._settings.processing_worker_enabled,
            running=bool(thread and thread.is_alive()),
            accepting_claims=accepting,
            active_run_id=active,
            pending_count=pending,
            worker_instance_id=self._worker_instance_id,
        )

    def stop(self) -> None:
        """Stop new claims and wait for an active operation to reach a safe boundary."""

        with self._state_lock:
            self._accepting_claims = False
            thread = self._thread
        self._stop_event.set()
        self._wake_event.set()
        if thread is None or not thread.is_alive():
            return
        thread.join(timeout=self._settings.processing_shutdown_timeout_seconds)
        if thread.is_alive():
            logger.warning(
                "Worker shutdown timed out with active_run_id=%s; startup recovery will reconcile it",
                self._active_run_id,
            )
        else:
            logger.info("Processing worker stopped")

    def _stop_requested(self) -> bool:
        return self._stop_event.is_set()

    def _claim_next(self) -> ClaimedJob | None:
        if self._stop_event.is_set():
            return None
        claimed_at = datetime.now(timezone.utc)
        with SessionLocal.begin() as session:
            run = session.scalar(
                select(ProcessingRun)
                .where(ProcessingRun.status == ProcessingStatus.QUEUED.value)
                .order_by(ProcessingRun.id)
                .with_for_update(skip_locked=True)
                .limit(1)
            )
            if run is None:
                return None
            background = _background_metadata(run)
            background.update(
                {
                    "implementation": "postgres_polling_single_worker_v1",
                    "worker_instance_id": self._worker_instance_id,
                    "claimed_at": claimed_at.isoformat(),
                    "claim_count": int(background.get("claim_count", 0)) + 1,
                }
            )
            background.setdefault("queued_at", claimed_at.isoformat())
            request_id = background.get("request_id")
            run.status = ProcessingStatus.RUNNING.value
            run.started_at = claimed_at
            run.completed_at = None
            run.duration_ms = None
            run.error_message = None
            _persist_background(run, background)
            run_id = run.id
        logger.info(
            "request_id=%s claimed_run_id=%s worker_instance_id=%s",
            request_id if isinstance(request_id, str) else "unavailable",
            run_id,
            self._worker_instance_id,
        )
        return ClaimedJob(
            run_id=run_id,
            request_id=request_id if isinstance(request_id, str) else None,
        )

    def _release_claim(self, job: ClaimedJob) -> None:
        """Return a claimed-but-not-started job to the durable queue."""

        with SessionLocal.begin() as session:
            run = session.get(ProcessingRun, job.run_id, with_for_update=True)
            if run is None or run.status != ProcessingStatus.RUNNING.value:
                return
            background = _background_metadata(run)
            if background.get("work_started_at"):
                return
            background["released_at_shutdown"] = datetime.now(timezone.utc).isoformat()
            background.pop("worker_instance_id", None)
            run.status = ProcessingStatus.QUEUED.value
            run.started_at = None
            _persist_background(run, background)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            job = self._claim_next()
            if job is None:
                self._wake_event.wait(self._settings.processing_worker_poll_seconds)
                self._wake_event.clear()
                continue
            if self._stop_event.is_set():
                self._release_claim(job)
                break
            with self._state_lock:
                self._active_run_id = job.run_id
            try:
                execute_processing_run(job.run_id, stop_requested=self._stop_requested)
            except Exception as exc:
                logger.exception("Unhandled worker failure for run %s", job.run_id)
                self._mark_unhandled_failure(job.run_id, exc)
            finally:
                with self._state_lock:
                    self._active_run_id = None
        with self._state_lock:
            self._accepting_claims = False

    @staticmethod
    def _mark_unhandled_failure(run_id: uuid.UUID, exc: Exception) -> None:
        with SessionLocal.begin() as session:
            run = session.get(ProcessingRun, run_id, with_for_update=True)
            if run is None or run.status == ProcessingStatus.COMPLETED.value:
                return
            run.status = ProcessingStatus.FAILED.value
            run.stage = "failed"
            run.completed_at = datetime.now(timezone.utc)
            run.error_message = "Processing failed unexpectedly. Inspect server logs with the run ID."
            background = _background_metadata(run)
            background["failure"] = {
                "category": "processing_failure",
                "exception_type": type(exc).__name__,
                "failed_at": run.completed_at.isoformat(),
            }
            _persist_background(run, background)


_worker_lock = threading.Lock()
_worker: ProcessingWorker | None = None


def get_processing_worker(settings: Settings | None = None) -> ProcessingWorker:
    global _worker
    configured = settings or get_settings()
    with _worker_lock:
        if _worker is None:
            _worker = ProcessingWorker(configured)
        return _worker
