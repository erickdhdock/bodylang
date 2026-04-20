from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
from typing import Any

import httpx

from config import CONFIG

logger = logging.getLogger(__name__)


async def send_webhook(callback_url: str, payload: dict[str, Any]) -> bool:
    cfg = CONFIG.webhook
    if not cfg.enabled or not callback_url:
        return False
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}

    auth_token = CONFIG.api.auth_token
    if auth_token:
        sig = hmac.new(auth_token.encode("utf-8"), body, hashlib.sha256).hexdigest()
        headers["X-Bodylang-Signature"] = f"sha256={sig}"

    backoff = cfg.retry_backoff_sec
    async with httpx.AsyncClient(timeout=cfg.timeout_sec) as client:
        for attempt in range(cfg.retry_count + 1):
            try:
                resp = await client.post(callback_url, content=body, headers=headers)
                if 200 <= resp.status_code < 300:
                    logger.info(
                        "Webhook delivered to %s (status=%d, attempt=%d)",
                        callback_url, resp.status_code, attempt + 1,
                    )
                    return True
                logger.warning(
                    "Webhook %s returned %d, attempt %d",
                    callback_url, resp.status_code, attempt + 1,
                )
            except httpx.HTTPError as e:
                logger.warning("Webhook error (%s), attempt %d: %s", callback_url, attempt + 1, e)
            if attempt < cfg.retry_count:
                await asyncio.sleep(backoff * (2 ** attempt))
    logger.error("Webhook failed after %d attempts: %s", cfg.retry_count + 1, callback_url)
    return False
