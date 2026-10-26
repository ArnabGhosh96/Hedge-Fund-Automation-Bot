"""FastAPI dashboard and autonomous, fail-closed scan engine."""
import asyncio
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.requests import Request

import config
from ai_brain import get_ai_decision, invert_decision, load_learned_rules, shape_execution_levels
from auditor import load_rules_document, run_audit
from data_engine import fetch_correlated_asset_moves, fetch_h1_atr, fetch_multi_timeframe_data, initialize_mt5
from execution import close_position, execute_trade, get_open_positions, modify_position_stop, position_tick_prices
from memory_store import append_decision_memory, append_trade_memory, load_trade_memory, reconcile_closed_trades

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))
bot_state: dict[str, Any] = {"is_running": False, "interval": config.SCAN_INTERVAL_SECONDS, "risk_percent": config.DEFAULT_RISK_PERCENT,
    "equity": 0.0, "last_logic": "Waiting for scan", "last_confidence": 0, "last_signal": "HOLD", "last_executed_signal": None, "last_entry_price": None,
    "last_stop_loss": None, "last_take_profit": None, "trade_history": [], "open_positions": [], "learned_rules": [], "last_audit_at": None,
    "last_symbol": None, "last_scan_at": None, "last_error": None, "reconciled_count": 0, "daily_pnl_percent": None}
_engine_lock = asyncio.Lock()
_audit_lock = asyncio.Lock()
_reconcile_lock = asyncio.Lock()
_loop_wakeup = asyncio.Event()
_next_entry_scan = 0.0
_next_position_check = 0.0
_last_reconcile_monotonic = 0.0
# Per-candle decision cache: unchanged candles never trigger a repeat AI call.
_decision_cache: dict[str, dict[str, Any]] = {}
# Same-candle entry guard: one open per (symbol, candle, direction) prevents churn loops.
_bar_entries: set[tuple[str, str, str]] = set()
# Daily circuit breaker state: one equity baseline per UTC day.
_daily_guard: dict[str, Any] = {"utc_date": None, "baseline_equity": 0.0, "tripped": False}


def _daily_loss_guard(equity: float | None) -> None:
    """Stop the engine for the UTC day when drawdown from the day's baseline reaches the limit.

    The baseline is the first equity seen each UTC day, so a restart mid-day keeps
    the surviving reduction in play rather than granting a clean slate.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if _daily_guard["utc_date"] != today:
        _daily_guard.update(utc_date=today, baseline_equity=equity or 0.0, tripped=False)
    limit: float = config.DAILY_LOSS_STOP_PERCENT
    if _daily_guard["baseline_equity"] > 0 and equity:
        bot_state["daily_pnl_percent"] = round(
            100.0 * (equity - _daily_guard["baseline_equity"]) / _daily_guard["baseline_equity"], 2)
    if not limit or _daily_guard["tripped"] or not _daily_guard["baseline_equity"] or not equity:
        return
    loss_percent = 100.0 * (_daily_guard["baseline_equity"] - equity) / _daily_guard["baseline_equity"]
    if loss_percent >= limit:
        _daily_guard["tripped"] = True
        bot_state["is_running"] = False
        bot_state["last_error"] = f"Daily loss limit hit: -{loss_percent:.2f}% >= {limit:.0f}%; engine stopped for this UTC day"
        log.error("[SYSTEM] Daily loss limit hit (%.2f%% >= %.0f%%); engine stopped for this UTC day",
                  loss_percent, limit)


class ControlRequest(BaseModel):
    action: str
    interval: int | None = Field(default=None, ge=5, le=86400)
    risk_percent: float | None = Field(default=None, gt=0, le=config.MAX_RISK_PERCENT)


class CloseRequest(BaseModel):
    ticket: int


class AuditRequest(BaseModel):
    force: bool = False


def _read_mt5_snapshot() -> tuple[float | None, list[dict[str, Any]]]:
    import MetaTrader5 as mt5
    account = mt5.account_info()
    positions = get_open_positions()
    formatted = [{"ticket": int(p.ticket), "symbol": p.symbol, "side": "BUY" if p.type == mt5.POSITION_TYPE_BUY else "SELL",
        "volume": float(p.volume), "price_open": float(p.price_open), "current_price": float(p.price_current), "sl": float(p.sl), "tp": float(p.tp), "profit": float(p.profit)} for p in positions]
    return (float(account.equity) if account else None), formatted


async def _refresh_state() -> None:
    try:
        equity, positions = await asyncio.to_thread(_read_mt5_snapshot)
        if equity is not None:
            bot_state["equity"] = equity
        bot_state["open_positions"] = positions
    except Exception as exc:
        bot_state["last_error"] = str(exc)
    bot_state["trade_history"] = load_trade_memory()[-100:]
    doc = load_rules_document()
    bot_state["learned_rules"] = doc["rules"]
    bot_state["last_audit_at"] = doc.get("last_audit_at")


async def _audit(force: bool) -> dict[str, Any]:
    async with _audit_lock:
        result = await asyncio.to_thread(run_audit, force)
        await _refresh_state()
        return result


async def _reconcile_and_refresh(force: bool = False) -> int:
    """Reconcile MT5 closures promptly while avoiding a history query on every UI poll."""
    global _last_reconcile_monotonic
    async with _reconcile_lock:
        now = time.monotonic()
        if not force and now - _last_reconcile_monotonic < 5.0:
            return 0
        _last_reconcile_monotonic = now
        newly_closed = await asyncio.to_thread(reconcile_closed_trades)
        bot_state["reconciled_count"] += newly_closed
        if newly_closed:
            log.info("[TRADE] MT5 history reconciliation found %s newly closed trade(s)", newly_closed)
            await _audit(True)
        else:
            await _refresh_state()
        return newly_closed


def _bar_key(market: dict[str, Any]) -> str:
    """Identity of the current H1/D1 candles; unchanged value => reuse the cached decision."""
    return f"{market['h1_data'].get('bar_time')}/{market['daily_data'].get('bar_time')}"


async def _scan_symbol(symbol: str, correlated_snapshot: dict[str, dict[str, float]]) -> None:
    market = await asyncio.to_thread(fetch_multi_timeframe_data, symbol)
    market["correlated_moves"] = {k: v for k, v in correlated_snapshot.items() if k != symbol}
    bar_key = _bar_key(market)
    cached = _decision_cache.get(symbol)
    cache_hit = bool(cached and cached.get("bar_key") == bar_key)
    if cache_hit:
        decision = dict(cached["decision"])
    else:
        decision = await asyncio.to_thread(get_ai_decision, market, symbol)
        _decision_cache[symbol] = {"bar_key": bar_key, "decision": dict(decision)}
        # One log line per fresh decision so every symbol's reasoning is visible in the logs.
        model_signal = decision["signal"]
        executed_signal = invert_decision(decision, market)["signal"] if config.INVERT_DECISIONS else model_signal
        log.info("[SCAN] %s %s%s conf=%s %s", symbol, model_signal,
                 f" -> executes {executed_signal}" if executed_signal != model_signal else "",
                 decision["confidence_score"], str(decision["logic"]).replace("\n", " ")[:200])
        # Persist the MODEL's decision (reasoning + the exact market snapshot behind it) for
        # future prompts, later reviews, and audits — the cache and journal stay model-pure
        # even in reverse-execution mode; only the executed side flips afterwards.
        # Cached repeats are already covered.
        append_decision_memory({"at": datetime.now(timezone.utc).isoformat(), "symbol": symbol,
                                "signal": model_signal, "confidence": decision["confidence_score"],
                                "stop_loss": decision["stop_loss"], "take_profit": decision["take_profit"],
                                "bid": market["bid"], "ask": market["ask"], "spread": market["spread"],
                                "mid_price": round((market["bid"] + market["ask"]) / 2, 6),
                                "h1_bar_time": market["h1_data"].get("bar_time"),
                                "daily_bar_time": market["daily_data"].get("bar_time"),
                                "executed_signal": executed_signal, "inverted": executed_signal != model_signal,
                                "logic": decision["logic"], "market": market})
        for stale in [e for e in _bar_entries if e[0] == symbol and e[1] != bar_key]:
            _bar_entries.discard(stale)
    # Reverse-execution mode: flip only the executed side (BUY->SELL, SELL->BUY; HOLD untouched).
    # The cached decision and the journal above stay model-pure, so prompt continuity and
    # audits keep seeing the model's own calls while the engine trades the inverse.
    model_signal = decision["signal"]
    if config.INVERT_DECISIONS:
        decision = invert_decision(decision, market)
    # Risk shaping on the executed side: stop snapped into its ATR band, then a fixed
    # reward multiple applied to the shaped stop. HOLD decisions are untouched.
    if decision["signal"] in {"BUY", "SELL"}:
        decision = shape_execution_levels(decision, market)
    bot_state.update(last_logic=decision["logic"], last_confidence=decision["confidence_score"], last_signal=model_signal,
                     last_executed_signal=decision["signal"], last_symbol=symbol,
                     last_scan_at=datetime.now(timezone.utc).isoformat(),
                     last_stop_loss=decision["stop_loss"], last_take_profit=decision["take_profit"])
    if cache_hit:
        bot_state["last_logic"] += " [CACHED this candle]"
    signal = decision["signal"]
    if signal not in {"BUY", "SELL"}:
        return
    positions = await asyncio.to_thread(get_open_positions, symbol)
    desired = signal
    same = [p for p in positions if ("BUY" if p.type == 0 else "SELL") == desired]
    if same:
        log.info("[DUPLICATE PREVENTED] %s already has %s exposure", symbol, desired)
        bot_state["last_logic"] += " [DUPLICATE PREVENTED] Existing same-direction position."
        return
    opposite = [p for p in positions if ("BUY" if p.type == 0 else "SELL") != desired]
    for position in opposite:
        log.info("[REVERSAL] Closing opposite %s position %s before new %s", symbol, position.ticket, desired)
        await asyncio.to_thread(close_position, position)
    if opposite:
        remaining = await asyncio.to_thread(get_open_positions, symbol)
        if remaining:
            raise RuntimeError(f"Reversal close not confirmed for {symbol}; entry withheld")
    entry_key = (symbol, bar_key, signal)
    if entry_key in _bar_entries:
        log.info("[SAME-BAR BLOCKED] %s %s already opened on the current candle", signal, symbol)
        bot_state["last_logic"] += " [SAME-BAR RE-ENTRY BLOCKED] Direction already opened on this candle."
        return
    fill = await asyncio.to_thread(execute_trade, symbol, signal, decision["stop_loss"], decision["take_profit"], bot_state["risk_percent"])
    _bar_entries.add(entry_key)
    bot_state["last_entry_price"] = fill["price"]
    append_trade_memory({"ticket": fill["order"], "deal": fill["deal"], "order": fill["order"], "position_id": fill["position_id"],
        "symbol": symbol, "side": signal, "entry_price": fill["price"], "stop_loss": fill["stop_loss"], "take_profit": fill["take_profit"],
        "volume": fill["volume"], "risk_percent": bot_state["risk_percent"], "confidence": decision["confidence_score"], "logic": decision["logic"],
        "market_context": {"h1_atr": market["h1_data"]["atr"], "daily_atr": market["daily_data"]["atr"],
            "h1_relative_volume": market["h1_data"]["relative_volume"], "daily_relative_volume": market["daily_data"]["relative_volume"],
            "h1_tick_volume": market["h1_data"]["tick_volume"], "daily_tick_volume": market["daily_data"]["tick_volume"], "correlated_moves": market["correlated_moves"]}})


def _exit_candidate_sl(position: Any, price: float, atr: float) -> float | None:
    """Tighten-only stop candidate for one open position: breakeven at +TRIGGER x R, then trail.

    Staged, unified logic: once price has moved BREAKEVEN_TRIGGER_R x original-risk in favor,
    the stop becomes eligible for entry-level (breakeven) and a running TRAIL_ATR_MULT x ATR
    level stays eligible thereafter (trailing remains armed once the stop has reached entry).
    The candidate is the most favorable of {current stop, entry, trail level}; the logic is
    stateless (restart-safe) and can never widen or lower an existing stop. Returns None when
    the improvement is under TRAIL_MIN_STEP_ATR x ATR, preventing SLTP-modify spam per pass.
    """
    is_buy = int(getattr(position, "type", -1)) == 0
    try:
        entry, current_sl = float(position.price_open), float(position.sl)
    except (TypeError, ValueError):
        return None
    if current_sl <= 0 or not atr or atr <= 0:
        return None
    candidate = current_sl
    remaining_risk = (entry - current_sl) if is_buy else (current_sl - entry)  # >0 before breakeven
    favorable_move = (price - entry) if is_buy else (entry - price)
    # Trailing arms once the breakeven trigger has fired (or was already passed / disabled).
    armed = remaining_risk <= 0 or config.BREAKEVEN_TRIGGER_R <= 0 or \
        favorable_move >= remaining_risk * config.BREAKEVEN_TRIGGER_R - 1e-12
    if not armed:
        return None
    if config.BREAKEVEN_TRIGGER_R > 0 and remaining_risk > 0:
        candidate = max(candidate, entry) if is_buy else min(candidate, entry)
    if config.TRAIL_ATR_MULT > 0:
        trail_level = price - config.TRAIL_ATR_MULT * atr if is_buy else price + config.TRAIL_ATR_MULT * atr
        candidate = max(candidate, trail_level) if is_buy else min(candidate, trail_level)
    improvement = (candidate - current_sl) if is_buy else (current_sl - candidate)
    return candidate if improvement >= config.TRAIL_MIN_STEP_ATR * atr else None


async def _manage_open_trade_exits() -> int:
    """Breakeven/ATR-trailing pass over this engine's open positions; returns modify count."""
    positions = await asyncio.to_thread(get_open_positions)
    managed = [p for p in positions if int(getattr(p, "magic", 0) or 0) == config.BOT_MAGIC]
    if not managed:
        return 0
    atr_by_symbol: dict[str, float] = {}
    changed = 0
    for position in managed:
        symbol = getattr(position, "symbol", None)
        if not symbol:
            continue
        atr = atr_by_symbol.get(symbol)
        if atr is None:
            atr = await asyncio.to_thread(fetch_h1_atr, symbol)
            atr_by_symbol[symbol] = atr or 0.0
        if not atr or atr <= 0:
            continue
        try:
            tick = await asyncio.to_thread(position_tick_prices, symbol)
            # The relevant side for a stop is the price you would exit at: bid for longs, ask for shorts.
            price = tick["bid"] if int(getattr(position, "type", -1)) == 0 else tick["ask"]
        except Exception as exc:
            log.warning("[EXITS] Tick unavailable for %s: %s", symbol, exc)
            continue
        candidate = _exit_candidate_sl(position, price, atr)
        if candidate is None:
            continue
        try:
            await asyncio.to_thread(modify_position_stop, position, candidate)
            changed += 1
        except Exception as exc:
            log.warning("[EXITS] Stop not moved for %s #%s: %s", symbol, getattr(position, "ticket", "?"), exc)
    return changed


async def trading_loop() -> None:
    global _next_entry_scan, _next_position_check
    while True:
        now = time.monotonic()
        # Daily circuit breaker: equity is refreshed every POSITION_MONITOR interval, so the
        # guard sees data at most a few minutes old. A mid-pass blowout trips at the next symbol.
        if bot_state["equity"]:
            _daily_loss_guard(bot_state["equity"])
        if bot_state["is_running"] and now >= _next_entry_scan:
            async with _engine_lock:
                # One shared cross-asset moves snapshot per cycle replaces per-symbol re-fetching.
                correlated_snapshot: dict[str, dict[str, float]] = {}
                try:
                    correlated_snapshot = await asyncio.to_thread(fetch_correlated_asset_moves, "", config.SYMBOLS)
                except Exception as exc:
                    log.warning("[SYSTEM] Correlated snapshot unavailable, continuing without it: %s", exc)
                for symbol in config.SYMBOLS:
                    if not bot_state["is_running"]:
                        break
                    if bot_state["equity"]:
                        _daily_loss_guard(bot_state["equity"])
                    try:
                        await _scan_symbol(symbol, correlated_snapshot)
                    except Exception as exc:
                        bot_state["last_error"] = f"{symbol}: {exc}"
                        log.exception("[SYSTEM] Scan failed for %s", symbol)
                        if "retcode=10026" in str(exc) or "AutoTrading disabled by server" in str(exc):
                            bot_state["is_running"] = False
                            log.error("[SYSTEM] Engine paused: MT5 server has disabled algorithmic trading")
                            break
            _next_entry_scan = time.monotonic() + bot_state["interval"]
        # Reconciliation and stop management share a cadence, independent of entry scans.
        if time.monotonic() >= _next_position_check:
            try:
                await _manage_open_trade_exits()
            except Exception as exc:
                log.exception("[EXITS] Stop management pass failed")
            try:
                await _reconcile_and_refresh(force=True)
            except Exception as exc:
                bot_state["last_error"] = f"Reconciliation: {exc}"
                log.exception("[SYSTEM] Reconciliation failed")
            _next_position_check = time.monotonic() + config.POSITION_MONITOR_INTERVAL_SECONDS

        deadlines = [_next_position_check]
        if bot_state["is_running"]:
            deadlines.append(_next_entry_scan)
        delay = max(0.05, min(deadlines) - time.monotonic())
        try:
            await asyncio.wait_for(_loop_wakeup.wait(), timeout=delay)
            _loop_wakeup.clear()
        except asyncio.TimeoutError:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await asyncio.to_thread(initialize_mt5)
        await _reconcile_and_refresh(force=True)
        await _refresh_state()
        doc = load_rules_document()
        bot_state["learned_rules"] = doc["rules"]
        if doc.get("last_audit_at"):
            try:
                from datetime import timedelta
                last = datetime.fromisoformat(doc["last_audit_at"].replace("Z", "+00:00"))
                if datetime.now(timezone.utc) - last >= timedelta(hours=config.AUDIT_COOLDOWN_HOURS):
                    asyncio.create_task(_audit(False))
            except (ValueError, TypeError):
                pass
        else:
            asyncio.create_task(_audit(False))
    except Exception as exc:
        bot_state["last_error"] = str(exc)
        log.exception("[SYSTEM] MT5 startup unavailable; API starts in disconnected, fail-closed mode")
    task = asyncio.create_task(trading_loop())
    log.info("[SYSTEM] Dashboard available on http://127.0.0.1:8000")
    yield
    bot_state["is_running"] = False
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    try:
        import MetaTrader5 as mt5
        mt5.shutdown()
    except Exception:
        pass


app = FastAPI(title="Autonomous Trading System", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # Render directly for compatibility with both legacy and current Starlette
    # TemplateResponse signatures; the dashboard template has no server context.
    return HTMLResponse(templates.get_template("index.html").render())


@app.get("/api/status")
async def status():
    await _reconcile_and_refresh()
    return bot_state


@app.post("/api/control")
async def control(payload: ControlRequest):
    global _next_entry_scan
    action = payload.action.lower()
    if action not in {"start", "stop"}:
        raise HTTPException(400, "action must be start or stop")
    if payload.interval is not None:
        bot_state["interval"] = payload.interval
    if payload.risk_percent is not None:
        bot_state["risk_percent"] = payload.risk_percent
    if action == "start":
        try:
            await asyncio.to_thread(initialize_mt5)
            bot_state["last_error"] = None
        except Exception as exc:
            bot_state["is_running"] = False
            bot_state["last_error"] = str(exc)
            raise HTTPException(503, f"MT5 is not connected: {exc}") from exc
        _next_entry_scan = 0.0
    bot_state["is_running"] = action == "start"
    _loop_wakeup.set()
    log.info("[SYSTEM] Engine %s interval=%ss risk=%s%%", action.upper(), bot_state["interval"], bot_state["risk_percent"])
    return {"ok": True, "is_running": bot_state["is_running"], "interval": bot_state["interval"], "risk_percent": bot_state["risk_percent"]}


@app.post("/api/close")
async def close(payload: CloseRequest):
    try:
        result = await asyncio.to_thread(close_position, payload.ticket)
        await _reconcile_and_refresh(force=True)
        return {"ok": True, **result}
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/audit")
async def audit(payload: AuditRequest):
    return await _audit(payload.force)


@app.get("/api/rules")
async def rules():
    return load_rules_document()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
