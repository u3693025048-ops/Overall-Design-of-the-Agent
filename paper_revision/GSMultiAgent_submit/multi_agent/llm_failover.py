#!/usr/bin/env python3
"""LLM endpoint failover when provider balance/quota is exhausted.

Primary endpoint comes from ``config.yaml`` ``llm``; optional ``llm.fallbacks``
are tried in order after a billing/balance/quota failure (HTTP 402, etc.).

Process-wide: once switched, ``get_config().llm`` fields and ``OPENAI_*`` env
vars are updated so all downstream clients pick up the new endpoint.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

_LOCK = threading.RLock()
_active_index: int = 0
_switched_reason: str = ""


@dataclass(frozen=True)
class LLMEndpoint:
    provider: str
    model: str
    base_url: str
    api_key: str
    name: str = ""

    @property
    def label(self) -> str:
        return self.name or self.model or self.base_url


_BALANCE_MARKERS = (
    "402",
    "insufficient balance",
    "insufficient_quota",
    "exceeded_current_quota",
    "account entitlement is exhausted",
    "suspended due to insufficient",
    "billing, credits, or account entitlement",
    "add credits or update billing",
    "payment required",
)


def is_balance_error(obj: Any) -> bool:
    """True if *obj* (exception / HTTP body / response text) indicates billing/balance failure."""
    if obj is None:
        return False
    text = str(obj).lower()
    if not text.strip():
        return False
    # Prefer explicit balance phrases over bare "402" alone when mixed with other noise,
    # but still treat HTTP 402 / status code 402 as balance.
    for m in _BALANCE_MARKERS:
        if m in text:
            return True
    code = getattr(obj, "status_code", None) or getattr(obj, "code", None)
    try:
        if int(code) == 402:
            return True
    except (TypeError, ValueError):
        pass
    return False


def endpoints_from_config() -> List[LLMEndpoint]:
    from multi_agent.config_loader import get_config

    cfg = get_config().llm
    primary = LLMEndpoint(
        provider=str(cfg.provider or "custom"),
        model=str(cfg.model or ""),
        base_url=str(cfg.base_url or ""),
        api_key=str(cfg.api_key or ""),
        name="primary",
    )
    out: List[LLMEndpoint] = [primary]
    for i, raw in enumerate(getattr(cfg, "fallbacks", None) or []):
        if not isinstance(raw, dict):
            continue
        key = str(raw.get("api_key") or "").strip()
        model = str(raw.get("model") or "").strip()
        base = str(raw.get("base_url") or "").strip()
        if not (key and model and base):
            continue
        out.append(
            LLMEndpoint(
                provider=str(raw.get("provider") or "custom"),
                model=model,
                base_url=base,
                api_key=key,
                name=str(raw.get("name") or f"fallback_{i+1}"),
            )
        )
    return out


def get_active_index() -> int:
    with _LOCK:
        return int(_active_index)


def get_active_endpoint() -> LLMEndpoint:
    eps = endpoints_from_config()
    with _LOCK:
        idx = min(max(0, _active_index), len(eps) - 1)
    return eps[idx]


def apply_endpoint(ep: LLMEndpoint, *, reason: str = "") -> None:
    """Push endpoint into env + live ``get_config().llm`` fields."""
    from multi_agent.config_loader import get_config

    if ep.base_url:
        os.environ["OPENAI_BASE_URL"] = ep.base_url
    if ep.api_key:
        os.environ["OPENAI_API_KEY"] = ep.api_key

    cfg = get_config().llm
    cfg.provider = ep.provider
    cfg.model = ep.model
    cfg.base_url = ep.base_url
    cfg.api_key = ep.api_key

    if reason:
        logger.warning(
            "[LLM-failover] active endpoint → %s (%s @ %s) reason=%s",
            ep.label,
            ep.model,
            ep.base_url,
            reason[:200],
        )
    else:
        logger.info(
            "[LLM-failover] active endpoint → %s (%s @ %s)",
            ep.label,
            ep.model,
            ep.base_url,
        )


def probe_endpoint(ep: LLMEndpoint, *, timeout: float = 25.0) -> Tuple[bool, str]:
    """Lightweight chat/completions probe. Returns (ok, detail)."""
    if not ep.api_key or not ep.base_url or not ep.model:
        return False, "missing api_key/base_url/model"
    url = ep.base_url.rstrip("/") + "/chat/completions"
    body = json.dumps(
        {
            "model": ep.model,
            "messages": [{"role": "user", "content": "ping reply with ok"}],
            "max_tokens": 8,
            "temperature": 0,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {ep.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", "replace")
            data = json.loads(raw) if raw else {}
            txt = (
                ((data.get("choices") or [{}])[0].get("message") or {}).get("content")
                or ""
            )
            return True, f"HTTP {resp.status} reply={txt!r}"
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:400]
        return False, f"HTTP {e.code} {detail}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def switch_to_next(*, reason: str = "") -> Optional[LLMEndpoint]:
    """Advance to the next configured endpoint. Returns new endpoint or None if exhausted."""
    global _active_index, _switched_reason
    eps = endpoints_from_config()
    with _LOCK:
        nxt = _active_index + 1
        if nxt >= len(eps):
            logger.error(
                "[LLM-failover] no more fallbacks after index=%s (%s)",
                _active_index,
                reason[:200],
            )
            return None
        _active_index = nxt
        _switched_reason = reason or _switched_reason
        ep = eps[_active_index]
    apply_endpoint(ep, reason=reason or "balance/quota")
    return ep


def note_failure_and_switch(err: Any) -> Optional[LLMEndpoint]:
    """If *err* looks like a balance/quota failure, switch to next endpoint."""
    if not is_balance_error(err):
        return None
    return switch_to_next(reason=str(err)[:300])


def ensure_usable_endpoint(*, probe: bool = True) -> LLMEndpoint:
    """Ensure the active endpoint works; probe+switch across the candidate list.

    Always applies the chosen endpoint to env/config before returning.
    """
    global _active_index
    eps = endpoints_from_config()
    if not eps:
        raise RuntimeError("no LLM endpoints configured")

    with _LOCK:
        start = min(max(0, _active_index), len(eps) - 1)

    last_detail = ""
    for idx in range(start, len(eps)):
        ep = eps[idx]
        if not probe:
            with _LOCK:
                _active_index = idx
            apply_endpoint(ep)
            return ep
        ok, detail = probe_endpoint(ep)
        last_detail = detail
        if ok:
            with _LOCK:
                _active_index = idx
            reason = "probe-ok" if idx == start else f"probe-failover; previous failed"
            apply_endpoint(ep, reason=reason)
            if idx > start:
                logger.warning(
                    "[LLM-failover] skipped dead endpoint(s) [%s..%s); using %s",
                    start,
                    idx,
                    ep.label,
                )
            return ep
        logger.warning(
            "[LLM-failover] probe failed for %s: %s",
            ep.label,
            detail[:300],
        )

    ep = eps[min(start, len(eps) - 1)]
    apply_endpoint(ep)
    logger.error("[LLM-failover] all endpoints failed last=%s", last_detail[:300])
    return ep


def openai_client_kwargs() -> Dict[str, Any]:
    ep = get_active_endpoint()
    apply_endpoint(ep)
    return {
        "api_key": ep.api_key or "sk-dummy",
        "base_url": ep.base_url or "https://api.openai.com/v1",
    }


async def chat_completion_create_with_failover(
    *,
    messages: Sequence[Dict[str, Any]],
    temperature: float,
    max_tokens: int,
    timeout: float = 60.0,
    max_retries_per_endpoint: int | None = None,
):
    """``chat.completions.create`` with automatic endpoint switch on balance errors.

    Transient errors (timeout / overloaded / generic 429) use exponential backoff
    with ±20% jitter (1s→2s→4s→…) and a finite retry cap. Billing/balance errors
    skip backoff and rotate to the next endpoint immediately.
    """
    global _active_index
    import asyncio

    import openai as _openai
    from multi_agent.llm_utils import (
        exponential_backoff_seconds,
        get_safe_temperature,
        llm_max_retries,
    )

    if max_retries_per_endpoint is None:
        max_retries_per_endpoint = llm_max_retries(5)

    eps = endpoints_from_config()
    start = get_active_index()
    last_exc: Optional[BaseException] = None

    for idx in range(start, len(eps)):
        with _LOCK:
            _active_index = idx
        ep = eps[idx]
        apply_endpoint(ep)
        client = _openai.AsyncOpenAI(
            api_key=ep.api_key or "sk-dummy",
            base_url=ep.base_url or "https://api.openai.com/v1",
            timeout=timeout,
            max_retries=0,
        )
        temp = get_safe_temperature(ep.model, temperature)
        for attempt in range(1, max_retries_per_endpoint + 1):
            try:
                return await client.chat.completions.create(
                    model=ep.model,
                    messages=list(messages),
                    temperature=temp,
                    max_tokens=max_tokens,
                )
            except BaseException as exc:
                last_exc = exc
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                if is_balance_error(exc):
                    logger.warning(
                        "[LLM-failover] chat balance error on %s (no backoff): %s",
                        ep.label,
                        exc,
                    )
                    break  # next endpoint immediately
                logger.warning(
                    "[LLM-failover] chat error on %s attempt %d/%d: %s",
                    ep.label,
                    attempt,
                    max_retries_per_endpoint,
                    exc,
                )
                if attempt >= max_retries_per_endpoint:
                    break  # try next endpoint if any
                delay = exponential_backoff_seconds(attempt)
                logger.info(
                    "[LLM-failover] backoff %.2fs before retry %d on %s",
                    delay,
                    attempt + 1,
                    ep.label,
                )
                await asyncio.sleep(delay)
    if last_exc:
        raise last_exc
    raise RuntimeError("chat_completion_create_with_failover: no endpoints")
