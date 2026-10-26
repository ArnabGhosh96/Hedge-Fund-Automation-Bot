"""LLM decision layer: deterministic pre-filters, bounded-retry requests, learned risk rules, strict validation."""
import json
import logging
import math
import time
from datetime import datetime, timezone
from typing import Any

import requests

import config
from memory_store import load_recent_decisions, realized_calibration

log = logging.getLogger(__name__)

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
# Approximate forex/commodity week gap in UTC: closes ~20:55 Friday, reopens ~21:05 Sunday.
# Broker server time usually sits UTC+2/+3, so these windows are intentionally approximate.
_FRIDAY_CLOSE_UTC_HOUR = 20 + 55 / 60
_SUNDAY_OPEN_UTC_HOUR = 21 + 5 / 60


def load_learned_rules(symbol: str) -> list[dict[str, Any]]:
    try:
        with open(config.RULES_FILE, "r", encoding="utf-8") as stream:
            rules = json.load(stream).get("rules", [])
        return [r for r in rules if r.get("active", True) and str(r.get("affected_symbol", "ALL")).upper() in {symbol.upper(), "ALL"}]
    except (OSError, json.JSONDecodeError, AttributeError):
        return []


def _fallback_hold(market_data: dict, reason: str) -> dict[str, Any]:
    price = (float(market_data.get("bid", 0)) + float(market_data.get("ask", 0))) / 2
    return {"signal": "HOLD", "confidence_score": 0, "stop_loss": price, "take_profit": price, "logic": reason}


def _prefilter_block_reason(market_data: dict[str, Any], symbol: str) -> str | None:
    """Deterministic, API-free gates evaluated before any LLM call. None means continue."""
    if config.SKIP_CLOSED_MARKETS and symbol.upper() not in config.ALWAYS_OPEN_SYMBOLS:
        now = datetime.now(timezone.utc)
        hour = now.hour + now.minute / 60
        if now.weekday() == 5 or (now.weekday() == 4 and hour >= _FRIDAY_CLOSE_UTC_HOUR) or (
                now.weekday() == 6 and hour < _SUNDAY_OPEN_UTC_HOUR):
            return "Skipped: forex/commodity markets are closed (weekend window)."
    spread = float(market_data.get("spread") or 0.0)
    h1 = market_data.get("h1_data") or {}
    atr = h1.get("atr")
    if atr and config.SKIP_SPREAD_ATR_RATIO > 0 and spread > float(atr) * config.SKIP_SPREAD_ATR_RATIO:
        return (f"Skipped: live spread {spread} exceeds {config.SKIP_SPREAD_ATR_RATIO:.0%} of H1 ATR "
                f"{float(atr)}; conditions are untradeable.")
    if config.SKIP_MIN_RELATIVE_VOLUME > 0 and h1.get("relative_volume") is not None:
        relative_volume = float(h1["relative_volume"])
        if relative_volume < config.SKIP_MIN_RELATIVE_VOLUME:
            return (f"Skipped: relative volume {relative_volume} below the {config.SKIP_MIN_RELATIVE_VOLUME} "
                    "liquidity floor.")
    return None


def _read_stream_content(response) -> str:
    """Consume an SSE chat stream and return the assembled assistant message content."""
    chunks: list[str] = []
    for raw_line in response.iter_lines():
        line = raw_line.decode("utf-8", "replace") if isinstance(raw_line, bytes) else str(raw_line)
        if not line or not line.startswith("data:"):
            continue
        event = line[len("data:"):].strip()
        if event == "[DONE]":
            break
        try:
            chunk = json.loads(event)
        except json.JSONDecodeError:
            continue  # skip keep-alive/unknown SSE events
        choices = chunk.get("choices") or []
        if choices:
            chunks.append(str((choices[0].get("delta") or {}).get("content") or ""))
    return "".join(chunks).strip()


def _request_decision_json(system: str, user_content: str, symbol: str) -> str:
    """POST one chat request with streaming, JSON mode and bounded retries; returns raw message content.

    Streaming matters here: the read timeout caps silence between bytes, not total generation
    time, so a 60s-long completion is fine as long as tokens keep arriving.
    """
    url = f"{config.DEEPSEEK_API_BASE}/chat/completions"
    timeout = (float(config.AI_CONNECT_TIMEOUT_SECONDS), float(config.AI_READ_TIMEOUT_SECONDS))
    allow_json_mode = bool(config.AI_JSON_MODE)
    streaming = bool(config.AI_STREAM)
    attempts = max(1, int(config.AI_MAX_ATTEMPTS))
    pause = max(0.5, float(config.AI_RETRY_BACKOFF_SECONDS))
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        body = {"model": config.DEEPSEEK_MODEL, "temperature": 0.1, "stream": streaming,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user_content}]}
        if allow_json_mode:
            body["response_format"] = {"type": "json_object"}
        try:
            response = requests.post(url, headers={"Authorization": f"Bearer {config.DEEPSEEK_API_KEY}"},
                                     json=body, stream=streaming, timeout=timeout)
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_error = exc
            log.warning("[AI] %s attempt %s/%s network error: %s", symbol, attempt, attempts, exc)
            if attempt < attempts:
                time.sleep(pause)
                continue
            raise
        if response.status_code == 400 and allow_json_mode and "response_format" in response.text.lower():
            allow_json_mode = False  # Gateway does not accept JSON mode; retry the call without it.
            log.warning("[AI] %s gateway rejected response_format; retrying without JSON mode", symbol)
            continue
        if response.status_code == 400 and streaming and "stream" in response.text.lower():
            streaming = False  # Gateway does not accept streaming; fall back to one blocking call.
            log.warning("[AI] %s gateway rejected streaming; retrying as a blocking call", symbol)
            continue
        if response.status_code in _RETRYABLE_STATUS and attempt < attempts:
            retry_after = response.headers.get("Retry-After")
            try:
                delay = min(10.0, max(pause, float(retry_after)))
            except (TypeError, ValueError):
                delay = pause
            log.warning("[AI] %s attempt %s/%s HTTP %s; retrying in %.1fs", symbol, attempt, attempts,
                        response.status_code, delay)
            last_error = requests.HTTPError(f"HTTP {response.status_code}", response=response)
            time.sleep(delay)
            continue
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        if streaming and response.status_code == 200 and ("event-stream" in content_type or "text/event" in content_type):
            content = _read_stream_content(response)
            if content:
                return content
            last_error = RuntimeError("AI stream ended without message content")
            if attempt < attempts:
                log.warning("[AI] %s attempt %s/%s stream had no content; retrying", symbol, attempt, attempts)
                time.sleep(pause)
                continue
            raise last_error
        # Non-streaming response (stream disabled or ignored by the gateway).
        return response.json()["choices"][0]["message"]["content"].strip()
    if last_error is not None:
        raise last_error
    raise RuntimeError("AI request exhausted retries without a response")


def invert_decision(decision: dict[str, Any], market_data: dict[str, Any]) -> dict[str, Any]:
    """Reverse-execution mode: a model BUY executes as SELL and a model SELL executes as BUY.

    HOLD passes through untouched. Stop/target distances from the original decision are
    mirrored around the executed side's reference price (ask for buys, bid for sells), so
    the flipped order keeps the same risk/reward geometry. The raw model call is preserved
    as model_signal and the logic text gains a REVERSED note for the audit trail.
    """
    raw_signal = str(decision.get("signal", "HOLD")).upper()
    if raw_signal not in {"BUY", "SELL"}:
        return decision
    flipped = "SELL" if raw_signal == "BUY" else "BUY"
    bid, ask = float(market_data["bid"]), float(market_data["ask"])
    try:
        atr = float((market_data.get("h1_data") or {}).get("atr") or 0.0)
    except (TypeError, ValueError):
        atr = 0.0
    anchor = ask if raw_signal == "BUY" else bid
    try:
        d_sl = abs(anchor - float(decision["stop_loss"])) if decision.get("stop_loss") is not None else 0.0
        d_tp = abs(float(decision["take_profit"]) - anchor) if decision.get("take_profit") is not None else 0.0
    except (TypeError, ValueError):
        d_sl = d_tp = 0.0
    d_sl = d_sl if d_sl > 0 else atr  # degenerate/missing levels fall back to a 1x H1 ATR cushion
    d_tp = d_tp if d_tp > 0 else atr
    reference = bid if flipped == "SELL" else ask
    if flipped == "SELL":  # model called BUY
        new_sl, new_tp = reference + d_sl, reference - d_tp
    else:                  # model called SELL
        new_sl, new_tp = reference - d_sl, reference + d_tp
    out = dict(decision)
    out.update(signal=flipped, stop_loss=new_sl, take_profit=new_tp, model_signal=raw_signal)
    out["logic"] = f"{decision.get('logic') or ''} [REVERSED EXECUTION: model called {raw_signal} -> engine executes {flipped}; SL/TP mirrored around {reference}]"
    return out


def shape_execution_levels(decision: dict[str, Any], market_data: dict[str, Any]) -> dict[str, Any]:
    """Enforce the risk geometry on the EXECUTED side of a BUY/SELL decision.

    1) The stop distance is clamped into [SL_MIN_ATR, SL_MAX_ATR] x H1 ATR (snap, never reject).
    2) When REWARD_RISK_MULTIPLE > 0, the target becomes MULTIPLE x the shaped stop distance,
       replacing the model's own target.
    Entry references are the actual fill sides (ask for BUY, bid for SELL). The model's raw
    levels survive on model_stop_loss/model_take_profit for the audit trail, and any applied
    adjustment is noted in the logic text.
    """
    signal = str(decision.get("signal", "HOLD")).upper()
    if signal not in {"BUY", "SELL"}:
        return decision
    try:
        atr = float((market_data.get("h1_data") or {}).get("atr") or 0.0)
    except (TypeError, ValueError):
        atr = 0.0
    reference = float(market_data["ask"]) if signal == "BUY" else float(market_data["bid"])
    shaped = dict(decision)
    shaped.update(model_stop_loss=decision.get("stop_loss"), model_take_profit=decision.get("take_profit"))
    try:
        stop_distance = abs(reference - float(decision["stop_loss"]))
    except (TypeError, ValueError, KeyError):
        return decision  # beyond saving; execute_trade's geometry validation will catch it
    clamped = stop_distance
    if atr > 0 and config.SL_MIN_ATR > 0:
        clamped = max(clamped, config.SL_MIN_ATR * atr)
    if atr > 0 and config.SL_MAX_ATR > 0:
        clamped = min(clamped, config.SL_MAX_ATR * atr)
    if clamped <= 0 or not (clamped > 0):
        return decision
    tolerance = (atr * 0.01) if atr > 0 else 1e-9
    notes: list[str] = []
    if abs(clamped - stop_distance) > tolerance:
        shaped["stop_loss"] = reference - clamped if signal == "BUY" else reference + clamped
        band = f"{clamped / atr:.2f} ATR" if atr > 0 else ""
        notes.append(f"SL snapped into band ({band} = {clamped:.5f} away)" if band else f"SL reshaped to {clamped:.5f}")
    if config.TRAIL_ONLY_MODE:
        # No broker target: the ATR trail (exit manager) is the only positive exit.
        shaped["take_profit"] = 0.0
        notes.append("TP suppressed — trail-only management")
    elif config.REWARD_RISK_MULTIPLE > 0:
        target_distance = config.REWARD_RISK_MULTIPLE * clamped
        shaped["take_profit"] = reference + target_distance if signal == "BUY" else reference - target_distance
        notes.append(f"TP set at {config.REWARD_RISK_MULTIPLE:g} x risk")
    if notes:
        shaped["logic"] = f"{decision.get('logic') or ''} [SHAPED: {'; '.join(notes)}]"
    return shaped


def _effective_confidence_threshold(symbol: str) -> int:
    """Entry gate adjusted by the calibration loop.

    A symbol whose closed-trade record shows a win-rate under CALIBRATION_MIN_WINRATE after
    CALIBRATION_MIN_TRADES closes must additionally clear
    CONFIDENCE_THRESHOLD + CALIBRATION_CONFIDENCE_PENALTY — engine-side, independent of what
    the model chose to report.
    """
    base = config.CONFIDENCE_THRESHOLD
    try:
        stats = (realized_calibration(symbol) or {}).get("symbol") or {}
    except Exception:
        stats = {}
    if (stats.get("closed") or 0) >= config.CALIBRATION_MIN_TRADES and \
            (stats.get("win_rate") or 0.0) < config.CALIBRATION_MIN_WINRATE:
        return min(100, base + config.CALIBRATION_CONFIDENCE_PENALTY)
    return base


# Appended to the standing system prompt at call time — explains the deterministic context
# fields (calibration dossier, key levels, session) the base instructions were written before.
CONTEXT_GUIDANCE = (
    " The snapshot carries deterministic additions the standing instructions predate. "
    "calibration reports this account's realized closed-trade performance: 'symbol' is THIS instrument's record "
    "(closed trades, wins, win_rate, avg_pnl) and 'confidence_bands' maps your reported confidence_score buckets to realized win rates across all symbols. "
    "Treat a losing current symbol record as direct evidence weighting against aggressive entries, and report a confidence that is consistent with that evidence "
    "rather than with enthusiasm; the engine also raises the entry gate for weak symbols, so the recorded numbers must be respected. "
    "h1_data.key_levels lists support/resistance clusters computed from H1 swing pivots (touches = how many pivots agree, distance_atr = distance from live price in ATRs); "
    "respect them: a BUY stop should sit beyond the nearest support cluster rather than inside it, and a target sitting just before an unbroken opposing level deserves reduced conviction. "
    "session_context names the UTC weekday and liquidity window (tokyo / london / overlap / new york / after-hours) — thin windows argue for conservatively expressed conviction."
)


def get_ai_decision(market_data: dict[str, Any], symbol: str) -> dict[str, Any]:
    block_reason = _prefilter_block_reason(market_data, symbol)
    if block_reason:
        return _fallback_hold(market_data, block_reason)
    if not config.DEEPSEEK_API_KEY:
        return _fallback_hold(market_data, "AI API key is not configured; fail-closed HOLD.")
    rules = load_learned_rules(symbol)
    recent_decisions = load_recent_decisions(symbol, config.RECENT_DECISIONS_IN_PROMPT)
    try:
        calibration = realized_calibration(symbol) if config.CALIBRATION_IN_PROMPT else None
    except Exception:
        calibration = None  # a read hiccup must never block a decision
    prompt_data = {"symbol": symbol, "bid": market_data["bid"], "ask": market_data["ask"], "spread": market_data["spread"],
                   "H1": market_data["h1_data"], "Daily": market_data["daily_data"],
                   "correlated_moves": market_data.get("correlated_moves") or {}, "session_context": market_data.get("session_context"),
                   "recent_decisions": recent_decisions, "active_rules": rules, "calibration": calibration}
    system = '''You are a risk-managed institutional quantitative trader on live FX, metals, energies and crypto. Return raw JSON only, exactly shaped as {"signal":"BUY"|"SELL"|"HOLD","confidence_score":0-100 integer,"stop_loss":number,"take_profit":number,"logic":"concise explanation including indicators and learned-rule deductions"}. Never invent data. Bias toward action on decisive setups: when a single timeframe's trend, adx strength, DI alignment, momentum and price action agree on a direction, say BUY or SELL and place a stop 1.0-1.5 ATR and a target 2.0-3.0 ATR from the live mid price; do not hold back a clean setup just to seem cautious. For BUY/SELL, confidence_score grades entry conviction: 70+ requires textbook alignment across both timeframes, 50-69 marks solid actionable setups, below 50 describes borderline ideas worth skipping. Reserve HOLD for genuine conflicts — mixed timeframes, fighting indicators, spread too wide for a stop this tight, no coherent setup — and in that case state in logic exactly which condition, if met first, would turn this into a trade. recent_decisions lists your own latest decisions for this symbol with their prices and brief reasoning: stay consistent with them when nothing material changed, and treat price merely reaching a stated invalidation or trigger level, a directional break in the candle series, or a shift in correlated_moves as the new evidence that may justify flipping your previous call. Apply the specified confidence_reduction_points whenever a learned setup condition matches, and explain the deduction. Stops and targets must be geometrically valid around the live mid price. The H1 and Daily blocks include a "recent" array of the last candles (t,o,h,l,c,v with t=UTC, newest last): read the actual price action from it — wick behaviour, swing rejection, and how indicators developed over the series — instead of judging only the latest snapshot values. correlated_moves reports 1-day and 5-day percent changes for the other symbols: use it for genuine cross-asset context (e.g. a USD-wide move across pairs, gold/oil/btc risk sentiment) before framing any trade. adx, di_plus and di_minus rate trend strength: in low-adx range-bound conditions, RSI extremes, structure breaks and momentum signals are unreliable, so prefer range-edge tactics or patience; distances such as close_vs_ema_atr and change_10_atr are in H1 ATR units — the same scale applies to every symbol.'''
    try:
        content = _request_decision_json(
            system + CONTEXT_GUIDANCE, "Analyze this snapshot and active risk rules: " + json.dumps(prompt_data, allow_nan=False), symbol)
        if content.startswith("```"):
            content = content.strip("`").removeprefix("json").strip()
        result = json.loads(content)
        signal = str(result.get("signal", "HOLD")).upper()
        confidence = result.get("confidence_score")
        sl, tp = float(result["stop_loss"]), float(result["take_profit"])
        if signal not in {"BUY", "SELL", "HOLD"} or isinstance(confidence, bool) or not isinstance(confidence, int) or not 0 <= confidence <= 100 or not math.isfinite(sl) or not math.isfinite(tp):
            raise ValueError("Decision JSON fields are invalid")
        price = (float(market_data["bid"]) + float(market_data["ask"])) / 2
        atr = float(market_data["h1_data"].get("atr") or 0)
        if signal in {"BUY", "SELL"} and not (atr > 0 and ((signal == "BUY" and sl < price < tp) or (signal == "SELL" and tp < price < sl))):
            sl = price - 1.5 * atr if signal == "BUY" else price + 1.5 * atr
            tp = price + 3.0 * atr if signal == "BUY" else price - 3.0 * atr
            result["logic"] = str(result.get("logic", "")) + " Invalid geometry corrected using H1 ATR."
        threshold = _effective_confidence_threshold(symbol)
        if confidence < threshold:
            signal = "HOLD"
            if threshold > config.CONFIDENCE_THRESHOLD:
                result["logic"] = str(result.get("logic", "")) + f" Calibration gate: weak {symbol} record requires >= {threshold}%."
        result.update(signal=signal, confidence_score=confidence, stop_loss=sl, take_profit=tp, logic=str(result.get("logic", ""))[:2000])
        return result
    except (requests.RequestException, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        log.exception("[AI] decision failed for %s", symbol)
        return _fallback_hold(market_data, f"AI decision unavailable: {exc}")
