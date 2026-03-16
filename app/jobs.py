from __future__ import annotations

import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from app.config import get_job_workers
from app.service import generate_loop_mp4_from_bytes_to_tempfile


JobStatus = Literal["queued", "running", "succeeded", "failed"]


@dataclass
class JobRecord:
    job_id: str
    status: JobStatus
    created_at: str
    started_at: str | None
    finished_at: str | None
    error: str | None
    out_path: str | None
    tmp_dir: str | None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobManager:
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=get_job_workers(), thread_name_prefix="gen2dlive-job")
        self._lock = threading.Lock()
        self._jobs: dict[str, JobRecord] = {}

    def create_job(
        self,
        *,
        image_bytes: bytes,
        duration_sec: float,
        fps: int,
        width: int | None,
        height: int | None,
        size: int | None,
        strength: float,
        particles: int,
    ) -> JobRecord:
        job_id = uuid.uuid4().hex
        record = JobRecord(
            job_id=job_id,
            status="queued",
            created_at=_utc_now_iso(),
            started_at=None,
            finished_at=None,
            error=None,
            out_path=None,
            tmp_dir=None,
        )
        with self._lock:
            self._jobs[job_id] = record

        self._executor.submit(
            self._run_job,
            job_id=job_id,
            image_bytes=image_bytes,
            duration_sec=duration_sec,
            fps=fps,
            width=width,
            height=height,
            size=size,
            strength=strength,
            particles=particles,
        )
        return record

    def get_job(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def consume_result(self, job_id: str) -> tuple[str, str] | None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None or record.status != "succeeded" or not record.out_path or not record.tmp_dir:
                return None
            out_path = record.out_path
            tmp_dir = record.tmp_dir
            record.out_path = None
            record.tmp_dir = None
            return out_path, tmp_dir

    def discard_job(self, job_id: str) -> None:
        with self._lock:
            record = self._jobs.pop(job_id, None)
        if record and record.tmp_dir:
            shutil.rmtree(record.tmp_dir, ignore_errors=True)

    def _run_job(
        self,
        *,
        job_id: str,
        image_bytes: bytes,
        duration_sec: float,
        fps: int,
        width: int | None,
        height: int | None,
        size: int | None,
        strength: float,
        particles: int,
    ) -> None:
        with self._lock:
            record = self._jobs[job_id]
            record.status = "running"
            record.started_at = _utc_now_iso()

        try:
            out_path, tmp_dir = generate_loop_mp4_from_bytes_to_tempfile(
                image_bytes=image_bytes,
                duration_sec=duration_sec,
                fps=fps,
                width=width,
                height=height,
                size=size,
                strength=strength,
                particles=particles,
            )
        except Exception as exc:
            with self._lock:
                record = self._jobs[job_id]
                record.status = "failed"
                record.error = str(exc)
                record.finished_at = _utc_now_iso()
            return

        with self._lock:
            record = self._jobs[job_id]
            record.status = "succeeded"
            record.finished_at = _utc_now_iso()
            record.out_path = out_path
            record.tmp_dir = tmp_dir


job_manager = JobManager()
