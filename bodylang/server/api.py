from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Header,
    Request,
    UploadFile,
)
from pydantic import BaseModel, Field

from config import CONFIG
from ingest import IngestionError, accept_local_file, job_workspace
from server.storage import JobState, get_state, put_state
from server.worker import enqueue, queue_depth, start_worker, stop_worker

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    await start_worker()
    logger.info("bodylang API started on %s:%d", CONFIG.api.host, CONFIG.api.port)
    yield
    await stop_worker()


app = FastAPI(title="bodylang", lifespan=_lifespan)


# ──────────────────────────────────────────────────────────────────────────────
# Auth
# ──────────────────────────────────────────────────────────────────────────────

def _check_auth(authorization: Optional[str]) -> None:
    expected = CONFIG.api.auth_token
    if not expected:
        return  # Auth disabled (dev mode).
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    if token != expected:
        raise HTTPException(status_code=403, detail="Invalid token")


# ──────────────────────────────────────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────────────────────────────────────

class AnalyzeUrlRequest(BaseModel):
    video_url: str = Field(min_length=1)
    callback_url: Optional[str] = None
    job_id: Optional[str] = None


class AnalyzeAccepted(BaseModel):
    job_id: str
    status: str
    estimated_seconds: int


class ResultResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    stage: str
    error: Optional[str] = None
    report: Optional[dict] = None


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/healthz")
async def healthz() -> dict:
    return {
        "ok": True,
        "queue_depth": queue_depth(),
        "analysis_fps": CONFIG.processing.analysis_fps,
        "pose_variant": CONFIG.processing.pose_variant,
        "gpu_delegate_requested": CONFIG.processing.gpu_delegate,
    }


@app.post("/analyze", status_code=202, response_model=AnalyzeAccepted)
async def analyze_url(
    req: AnalyzeUrlRequest,
    authorization: Optional[str] = Header(default=None),
) -> AnalyzeAccepted:
    _check_auth(authorization)
    job_id = req.job_id or uuid.uuid4().hex
    if get_state(job_id) is not None:
        raise HTTPException(status_code=409, detail="job_id already exists")
    state = JobState(
        job_id=job_id,
        source="url",
        video_url=req.video_url,
        callback_url=req.callback_url,
        created_at=time.time(),
    )
    put_state(state)
    await enqueue(job_id)
    return AnalyzeAccepted(
        job_id=job_id,
        status="queued",
        estimated_seconds=_estimate_runtime(),
    )


@app.post("/analyze/upload", status_code=202, response_model=AnalyzeAccepted)
async def analyze_upload(
    file: UploadFile = File(...),
    callback_url: Optional[str] = Form(default=None),
    job_id: Optional[str] = Form(default=None),
    authorization: Optional[str] = Header(default=None),
) -> AnalyzeAccepted:
    _check_auth(authorization)
    if (
        file.content_type
        and file.content_type not in CONFIG.ingestion.allowed_content_types
    ):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported content type: {file.content_type}",
        )
    job_id = job_id or uuid.uuid4().hex
    if get_state(job_id) is not None:
        raise HTTPException(status_code=409, detail="job_id already exists")

    ws = job_workspace(job_id)
    dst = ws / (file.filename or "upload.mp4")
    max_bytes = CONFIG.ingestion.max_size_mb * 1024 * 1024
    written = 0
    with open(dst, "wb") as f:
        while True:
            chunk = await file.read(1 << 20)
            if not chunk:
                break
            written += len(chunk)
            if written > max_bytes:
                f.close()
                dst.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="Upload too large")
            f.write(chunk)

    state = JobState(
        job_id=job_id,
        source="upload",
        video_path=str(dst),
        callback_url=callback_url,
        created_at=time.time(),
    )
    put_state(state)
    await enqueue(job_id)
    return AnalyzeAccepted(
        job_id=job_id,
        status="queued",
        estimated_seconds=_estimate_runtime(),
    )


@app.get("/result/{job_id}", response_model=ResultResponse)
async def get_result(
    job_id: str,
    authorization: Optional[str] = Header(default=None),
) -> ResultResponse:
    _check_auth(authorization)
    state = get_state(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return ResultResponse(
        job_id=state.job_id,
        status=state.status,
        progress=state.progress,
        stage=state.stage,
        error=state.error,
        report=state.report.model_dump() if state.report else None,
    )


def _estimate_runtime() -> int:
    # Very rough: assume ~6 min for a 30 min clip on GPU, scale with queue.
    base_seconds = 360
    return base_seconds * (queue_depth() + 1)


# ──────────────────────────────────────────────────────────────────────────────
# Entrypoint (uvicorn server.api:app)
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import uvicorn

    uvicorn.run(
        "server.api:app",
        host=CONFIG.api.host,
        port=CONFIG.api.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
