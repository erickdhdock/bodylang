from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

import httpx

from config import CONFIG
from frames import VideoProbe, probe

logger = logging.getLogger(__name__)


class IngestionError(Exception):
    pass


def _ensure_temp_dir() -> Path:
    root = Path(CONFIG.ingestion.temp_dir)
    root.mkdir(parents=True, exist_ok=True)
    return root


def job_workspace(job_id: str) -> Path:
    ws = _ensure_temp_dir() / job_id
    ws.mkdir(parents=True, exist_ok=True)
    return ws


def accept_local_file(src_path: str | Path, job_id: str) -> Path:
    """Copy an already-local file into the job workspace."""
    src = Path(src_path)
    if not src.exists():
        raise IngestionError(f"Source file not found: {src}")
    _validate_size(src)
    dst = job_workspace(job_id) / src.name
    shutil.copy2(src, dst)
    return dst


def download_from_url(url: str, job_id: str) -> Path:
    """Stream-download a video from a URL into the job workspace."""
    ws = job_workspace(job_id)
    suffix = _infer_suffix(url) or ".mp4"
    dst = ws / f"source{suffix}"
    logger.info("Downloading %s -> %s", url, dst)
    max_bytes = CONFIG.ingestion.max_size_mb * 1024 * 1024
    written = 0
    timeout = CONFIG.ingestion.url_download_timeout_sec
    with httpx.stream(
        "GET", url, timeout=timeout, follow_redirects=True
    ) as response:
        response.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in response.iter_bytes(chunk_size=1 << 20):
                if not chunk:
                    continue
                written += len(chunk)
                if written > max_bytes:
                    f.close()
                    os.unlink(dst)
                    raise IngestionError(
                        f"Download exceeds max_size_mb={CONFIG.ingestion.max_size_mb}"
                    )
                f.write(chunk)
    return dst


def _infer_suffix(url: str) -> str:
    path = url.split("?", 1)[0]
    for ext in (".mp4", ".mov", ".webm", ".mkv"):
        if path.lower().endswith(ext):
            return ext
    return ""


def _validate_size(path: Path) -> None:
    max_bytes = CONFIG.ingestion.max_size_mb * 1024 * 1024
    size = path.stat().st_size
    if size > max_bytes:
        raise IngestionError(
            f"File size {size / 1024 / 1024:.1f}MB exceeds "
            f"max_size_mb={CONFIG.ingestion.max_size_mb}"
        )


def validate_video(path: Path) -> VideoProbe:
    """Probe the video and reject if it violates duration limits."""
    info = probe(path)
    if info.duration_sec > CONFIG.ingestion.max_duration_min * 60:
        raise IngestionError(
            f"Video duration {info.duration_sec:.0f}s exceeds "
            f"max_duration_min={CONFIG.ingestion.max_duration_min}"
        )
    if info.duration_sec <= 0:
        raise IngestionError("Video duration is zero or unreadable")
    return info


def cleanup_workspace(job_id: str) -> None:
    ws = Path(CONFIG.ingestion.temp_dir) / job_id
    if ws.exists():
        shutil.rmtree(ws, ignore_errors=True)
