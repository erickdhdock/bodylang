from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from config import CONFIG
from schema import BodyLangReport

logger = logging.getLogger(__name__)


@dataclass
class JobState:
    job_id: str
    status: str = "queued"            # queued | running | done | failed
    progress: float = 0.0
    stage: str = "queued"
    error: Optional[str] = None
    callback_url: Optional[str] = None
    source: str = ""                   # "upload" or "url"
    video_url: Optional[str] = None
    video_path: Optional[str] = None
    report: Optional[BodyLangReport] = None
    created_at: float = 0.0
    finished_at: Optional[float] = None


_STATE: dict[str, JobState] = {}


def get_state(job_id: str) -> Optional[JobState]:
    return _STATE.get(job_id)


def put_state(state: JobState) -> None:
    _STATE[state.job_id] = state


def all_states() -> list[JobState]:
    return list(_STATE.values())


def results_dir() -> Path:
    d = Path(CONFIG.api.storage_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d


def persist_report(report: BodyLangReport) -> Path:
    out = results_dir() / report.job_id
    out.mkdir(parents=True, exist_ok=True)
    path = out / "report.json"
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    logger.info("[%s] report persisted: %s", report.job_id, path)
    return path


def load_report(job_id: str) -> Optional[BodyLangReport]:
    path = results_dir() / job_id / "report.json"
    if not path.exists():
        return None
    return BodyLangReport.model_validate_json(path.read_text(encoding="utf-8"))
