from __future__ import annotations

import asyncio
import logging
import time
import traceback
from pathlib import Path

from config import CONFIG
from ingest import cleanup_workspace, download_from_url, validate_video
from pipeline import BodyLangPipeline, PipelineError
from server.callback import send_webhook
from server.storage import JobState, get_state, persist_report, put_state

logger = logging.getLogger(__name__)


_QUEUE: asyncio.Queue[str] = asyncio.Queue()
_TASK: asyncio.Task | None = None


async def enqueue(job_id: str) -> None:
    await _QUEUE.put(job_id)


def queue_depth() -> int:
    return _QUEUE.qsize()


async def start_worker() -> None:
    global _TASK
    if _TASK is None or _TASK.done():
        _TASK = asyncio.create_task(_run_forever(), name="bodylang-worker")


async def stop_worker() -> None:
    global _TASK
    if _TASK is not None:
        _TASK.cancel()
        try:
            await _TASK
        except asyncio.CancelledError:
            pass
        _TASK = None


async def _run_forever() -> None:
    pipeline = BodyLangPipeline()
    while True:
        job_id = await _QUEUE.get()
        try:
            await _process_one(pipeline, job_id)
        except Exception as e:  # noqa: BLE001
            logger.exception("Unhandled error processing job %s", job_id)
            state = get_state(job_id)
            if state is not None:
                state.status = "failed"
                state.error = str(e)
                state.finished_at = time.time()
                put_state(state)
        finally:
            _QUEUE.task_done()


async def _process_one(pipeline: BodyLangPipeline, job_id: str) -> None:
    state = get_state(job_id)
    if state is None:
        logger.warning("No state for job %s, dropping", job_id)
        return

    state.status = "running"
    state.stage = "ingest"
    state.progress = 0.01
    put_state(state)

    loop = asyncio.get_running_loop()

    try:
        # 1. Materialize video locally if we were given a URL.
        if state.source == "url":
            assert state.video_url
            state.stage = "download"
            put_state(state)
            video_path = await loop.run_in_executor(
                None, download_from_url, state.video_url, job_id
            )
            state.video_path = str(video_path)
            put_state(state)

        video_path = Path(state.video_path or "")
        if not video_path.exists():
            raise PipelineError("Video file missing before pipeline start.")

        # 2. Validate duration/size.
        await loop.run_in_executor(None, validate_video, video_path)

        # 3. Run pipeline on executor (CPU/GPU-bound, blocks event loop).
        def _progress(pct: float, stage: str) -> None:
            s = get_state(job_id)
            if s is not None:
                s.progress = pct
                s.stage = stage
                put_state(s)

        report = await loop.run_in_executor(
            None, pipeline.run, video_path, job_id, _progress
        )

        await loop.run_in_executor(None, persist_report, report)
        state.report = report
        state.status = "done"
        state.progress = 1.0
        state.stage = "done"
    except (PipelineError, Exception) as e:  # noqa: BLE001
        logger.error("[%s] pipeline failed: %s\n%s", job_id, e, traceback.format_exc())
        state.status = "failed"
        state.error = str(e)
    finally:
        state.finished_at = time.time()
        put_state(state)
        # Always clean up raw video to free disk.
        cleanup_workspace(job_id)

    # 4. Deliver webhook (non-blocking, but we await here since worker is already
    #    serial per-job).
    if state.callback_url and CONFIG.webhook.enabled:
        payload = {
            "job_id": job_id,
            "status": state.status,
            "error": state.error,
            "report": state.report.model_dump() if state.report else None,
        }
        await send_webhook(state.callback_url, payload)
