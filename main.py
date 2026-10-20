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
from ai_brain import get_ai_decision, get_ai_exit_decision, load_learned_rules
from auditor import load_rules_document, run_audit
from data_engine import fetch_correlated_asset_prices, fetch_multi_timeframe_data, initialize_mt5
from execution import close_position, execute_trade, get_open_positions, modify_position_protection
from memory_store import append_management_event, append_trade_memory, load_trade_memory, reconcile_closed_trades

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))
bot_state: dict[str, Any] = {"is_running": False, "interval": config.SCAN_INTERVAL_SECONDS, "risk_percent": config.DEFAULT_RISK_PERCENT,
    "equity": 0.0, "last_logic": "Waiting for scan", "last_confidence": 0, "last_signal": "HOLD", "last_entry_price": None,
    "last_stop_loss": None, "last_take_profit": None, "trade_history": [], "open_positions": [], "learned_rules": [], "last_audit_at": None,
    "last_symbol": None, "last_scan_at": None, "last_error": None, "reconciled_count": 0}
bot_state.update(last_exit_logic="Position manager is monitoring open MT5 positions.", last_exit_review_at=None, exit_reviews=[])
bot_state.update(exit_confidence_threshold=config.EXIT_CONFIDENCE_THRESHOLD, trail_activate_atr=config.TRAIL_ACTIVATE_ATR,
                 trail_distance_atr=config.TRAIL_DISTANCE_ATR, exit_ai_review_seconds=config.EXIT_AI_REVIEW_SECONDS,
                 position_monitor_interval=config.POSITION_MONITOR_INTERVAL_SECONDS)
_engine_lock = asyncio.Lock()
_audit_lock = asyncio.Lock()
_reconcile_lock = asyncio.Lock()
_loop_wakeup = asyncio.Event()
_last_exit_ai_review: dict[int, float] = {}
_exit_reviews_by_ticket: dict[int, dict[str, Any]] = {}
_next_entry_scan = 0.0
_next_position_check = 0.0
_last_reconcile_monotonic = 0.0


class ControlRequest(BaseModel):
    action: str
    interval: int | None = Field(default=None, ge=5, le=86400)
    risk_percent: float | None = Field(default=None, gt=0, le=config.MAX_RISK_PERCENT)


class CloseRequest(BaseModel):
    ticket: int


class AuditRequest(BaseModel):
    force: bool = False


def _record_exit_review(position: Any, action: str, confidence: int | None, logic: str,
                        stop_loss: float | None = None, take_profit: float | None = None,
                        reviewed_at: str | None = None) -> None:
    ticket = int(position.ticket)
    previous = _exit_reviews_by_ticket.get(ticket, {})
    _exit_reviews_by_ticket[ticket] = {
        "ticket": ticket, "symbol": str(position.symbol),
        "side": "BUY" if int(position.type) == 0 else "SELL",
        "action": action, "confidence_score": confidence, "logic": logic,
        "stop_loss": stop_loss if stop_loss is not None else (float(position.sl) if position.sl else None),
        "take_profit": take_profit if take_profit is not None else (float(position.tp) if position.tp else None),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "last_reviewed_at": reviewed_at or previous.get("last_reviewed_at"),
    }
    bot_state["exit_reviews"] = sorted(_exit_reviews_by_ticket.values(), key=lambda row: (row["symbol"], row["ticket"]))


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
    for position in bot_state.get("open_positions", []):
        ticket = int(position["ticket"])
        if ticket in _exit_reviews_by_ticket:
            continue
        matching = next((record for record in reversed(bot_state["trade_history"])
                         if ticket in {int(record.get(key) or 0) for key in ("position_id", "ticket", "order")}), None)
        events = matching.get("management_events", []) if matching else []
        previous = next((event for event in reversed(events) if event.get("action") == "AI_REVIEW"), None)
        if previous:
            _exit_reviews_by_ticket[ticket] = {
                "ticket": ticket, "symbol": position["symbol"], "side": position["side"],
                "action": previous.get("decision_action", "HOLD"),
                "confidence_score": previous.get("confidence"), "logic": previous.get("reason", ""),
                "stop_loss": position.get("sl"), "take_profit": position.get("tp"),
                "updated_at": previous.get("at"), "last_reviewed_at": previous.get("at"),
            }
    bot_state["exit_reviews"] = sorted(_exit_reviews_by_ticket.values(), key=lambda row: (row["symbol"], row["ticket"]))
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


async def _scan_symbol(symbol: str) -> None:
    market = await asyncio.to_thread(fetch_multi_timeframe_data, symbol)
    market["correlated_prices"] = await asyncio.to_thread(fetch_correlated_asset_prices, symbol, config.SYMBOLS)
    decision = await asyncio.to_thread(get_ai_decision, market, symbol)
    bot_state.update(last_logic=decision["logic"], last_confidence=decision["confidence_score"], last_signal=decision["signal"],
                     last_symbol=symbol, last_scan_at=datetime.now(timezone.utc).isoformat(), last_stop_loss=decision["stop_loss"], last_take_profit=decision["take_profit"])
    signal = decision["signal"]
    if signal not in {"BUY", "SELL"}:
        return
    positions = await asyncio.to_thread(get_open_positions, symbol)
    desired = signal
    same = [p for p in positions if ("BUY" if p.type == 0 else "SELL") == desired]
    opposite = [p for p in positions if ("BUY" if p.type == 0 else "SELL") != desired]
    if same:
        log.info("[DUPLICATE PREVENTED] %s already has %s exposure", symbol, desired)
        bot_state["last_logic"] += " [DUPLICATE PREVENTED] Existing same-direction position."
        return
    for position in opposite:
        log.info("[REVERSAL] Closing opposite %s position %s before new %s", symbol, position.ticket, desired)
        await asyncio.to_thread(close_position, position)
    # Confirm the close completed before opening exposure in the other direction.
    remaining = await asyncio.to_thread(get_open_positions, symbol)
    if remaining:
        raise RuntimeError(f"Reversal close not confirmed for {symbol}; entry withheld")
    fill = await asyncio.to_thread(execute_trade, symbol, signal, decision["stop_loss"], decision["take_profit"], bot_state["risk_percent"])
    bot_state["last_entry_price"] = fill["price"]
    append_trade_memory({"ticket": fill["order"], "deal": fill["deal"], "order": fill["order"], "position_id": fill["position_id"],
        "symbol": symbol, "side": signal, "entry_price": fill["price"], "stop_loss": fill["stop_loss"], "take_profit": fill["take_profit"],
        "volume": fill["volume"], "risk_percent": bot_state["risk_percent"], "confidence": decision["confidence_score"], "logic": decision["logic"],
        "market_context": {"h1_atr": market["h1_data"]["atr"], "daily_atr": market["daily_data"]["atr"],
            "h1_relative_volume": market["h1_data"]["relative_volume"], "daily_relative_volume": market["daily_data"]["relative_volume"],
            "h1_tick_volume": market["h1_data"]["tick_volume"], "daily_tick_volume": market["daily_data"]["tick_volume"], "correlated_prices": market["correlated_prices"]}})


async def _manage_open_positions() -> None:
    """Review every live position, trail protection, and apply constrained AI exits/updates."""
    try:
        positions = await asyncio.to_thread(get_open_positions)
    except Exception as exc:
        bot_state["last_exit_logic"] = f"Could not read open positions: {exc}"
        log.warning("[TRADE] Position manager could not read positions: %s", exc)
        return
    if not positions:
        _last_exit_ai_review.clear()
        _exit_reviews_by_ticket.clear()
        bot_state["exit_reviews"] = []
        bot_state["last_exit_logic"] = "No open positions to manage."
        return
    try:
        import MetaTrader5 as mt5
    except ImportError:
        bot_state["last_exit_logic"] = "MT5 API unavailable; broker-side SL/TP remains the only protection."
        for position in positions:
            _record_exit_review(position, "ERROR", None, bot_state["last_exit_logic"])
        return

    live_tickets = {int(p.ticket) for p in positions}
    for stale_ticket in set(_last_exit_ai_review) - live_tickets:
        _last_exit_ai_review.pop(stale_ticket, None)
    for stale_ticket in set(_exit_reviews_by_ticket) - live_tickets:
        _exit_reviews_by_ticket.pop(stale_ticket, None)
    bot_state["exit_reviews"] = sorted(_exit_reviews_by_ticket.values(), key=lambda row: (row["symbol"], row["ticket"]))

    for position in positions:
        ticket, symbol = int(position.ticket), str(position.symbol)
        side = "BUY" if position.type == mt5.POSITION_TYPE_BUY else "SELL"
        if ticket not in _exit_reviews_by_ticket:
            _record_exit_review(position, "MONITORING", None, "No AI review is recorded yet; waiting for the first review.")
        try:
            market = await asyncio.to_thread(fetch_multi_timeframe_data, symbol)
            bid, ask = float(market["bid"]), float(market["ask"])
            atr = float(market["h1_data"].get("atr") or 0)
            entry = float(position.price_open)
            current_sl, current_tp = float(position.sl or 0), float(position.tp or 0)
            if atr <= 0 or not (bid > 0 and ask >= bid):
                message = f"Invalid quote/ATR; keeping existing broker protection."
                bot_state["last_exit_logic"] = f"{symbol} #{ticket}: {message}"
                _record_exit_review(position, "MONITORING", None, message)
                continue

            is_buy = side == "BUY"
            market_price = bid if is_buy else ask
            favorable = (market_price - entry) if is_buy else (entry - market_price)
            # If a manual/external position has no SL/TP, install conservative ATR brackets.
            planned_sl = current_sl or (entry - 1.5 * atr if is_buy else entry + 1.5 * atr)
            planned_tp = current_tp or (entry + 3.0 * atr if is_buy else entry - 3.0 * atr)
            if favorable >= config.TRAIL_ACTIVATE_ATR * atr:
                trailing = market_price - config.TRAIL_DISTANCE_ATR * atr if is_buy else market_price + config.TRAIL_DISTANCE_ATR * atr
                # At activation, at least bring the stop to entry; thereafter trail by ATR.
                trailing = max(trailing, entry) if is_buy else min(trailing, entry)
                planned_sl = max(planned_sl, trailing) if is_buy else min(planned_sl, trailing)

            # If an unprotected position has already crossed its emergency ATR stop/target,
            # request a market close rather than installing an invalid stop behind price.
            emergency_stop = entry - 1.5 * atr if is_buy else entry + 1.5 * atr
            missing_sl_breached = current_sl <= 0 and ((is_buy and bid <= emergency_stop) or (not is_buy and ask >= emergency_stop))
            missing_tp_reached = current_tp <= 0 and ((is_buy and bid >= planned_tp) or (not is_buy and ask <= planned_tp))
            if missing_sl_breached or missing_tp_reached:
                reason = "unprotected position crossed ATR stop" if missing_sl_breached else "unprotected position reached ATR target"
                result = await asyncio.to_thread(close_position, position)
                append_management_event(ticket, {"action": "CLOSE", "reason": reason, "deal": result["deal"], "price": result["price"]})
                bot_state["last_exit_logic"] = f"{symbol} #{ticket}: closed because an unprotected position crossed its {reason.split()[-1]} level."
                _record_exit_review(position, "CLOSED", None, reason, planned_sl, planned_tp)
                log.info("[TRADE] Closed %s #%s: %s", symbol, ticket, reason)
                continue

            now = time.monotonic()
            ai_decision = {"action": "HOLD", "confidence_score": 0, "stop_loss": None, "take_profit": None,
                           "logic": "AI review interval not due; ATR protection checked."}
            review_was_run = False
            if now - _last_exit_ai_review.get(ticket, 0.0) >= config.EXIT_AI_REVIEW_SECONDS:
                position_data = {"ticket": ticket, "symbol": symbol, "side": side, "volume": float(position.volume),
                    "entry_price": entry, "current_price": market_price, "bid": bid, "ask": ask, "floating_profit": float(position.profit),
                    "current_stop_loss": current_sl or None, "current_take_profit": current_tp or None,
                    "favorable_move_atr": favorable / atr}
                ai_decision = await asyncio.to_thread(get_ai_exit_decision, position_data, market)
                _last_exit_ai_review[ticket] = now
                review_was_run = True
                bot_state["last_exit_review_at"] = datetime.now(timezone.utc).isoformat()
                ai_decision["reviewed_at"] = bot_state["last_exit_review_at"]
                bot_state["last_exit_logic"] = f"{symbol} #{ticket} {ai_decision['action']} ({ai_decision['confidence_score']}%): {ai_decision['logic']}"
                append_management_event(ticket, {"action": "AI_REVIEW", "decision_action": ai_decision["action"],
                    "confidence": ai_decision["confidence_score"], "reason": ai_decision["logic"]})

            if not review_was_run:
                previous_review = _exit_reviews_by_ticket.get(ticket, {})
                ai_decision["display_action"] = previous_review.get("action", "MONITORING")
                ai_decision["display_confidence"] = previous_review.get("confidence_score")
                ai_decision["display_logic"] = previous_review.get("logic", "No AI review is recorded yet; waiting for the first review.")
            else:
                ai_decision["display_action"] = ai_decision["action"]
                ai_decision["display_confidence"] = ai_decision["confidence_score"]
                ai_decision["display_logic"] = ai_decision["logic"]
                _record_exit_review(position, ai_decision["action"], ai_decision["confidence_score"],
                                    ai_decision["logic"], planned_sl, planned_tp, ai_decision.get("reviewed_at"))

            if ai_decision["action"] == "CLOSE" and ai_decision["confidence_score"] >= config.EXIT_CONFIDENCE_THRESHOLD:
                result = await asyncio.to_thread(close_position, position)
                append_management_event(ticket, {"action": "AI_CLOSE", "confidence": ai_decision["confidence_score"],
                    "reason": ai_decision["logic"], "deal": result["deal"], "price": result["price"]})
                log.info("[TRADE] [AI EXIT] Closed %s #%s at %s: %s", symbol, ticket, result["price"], ai_decision["logic"])
                bot_state["last_exit_logic"] = f"{symbol} #{ticket}: closed by AI risk review ({ai_decision['confidence_score']}%). {ai_decision['logic']}"
                _record_exit_review(position, "CLOSED", ai_decision["confidence_score"], ai_decision["logic"], planned_sl, planned_tp)
                continue

            if ai_decision["action"] == "UPDATE" and ai_decision["confidence_score"] >= config.EXIT_CONFIDENCE_THRESHOLD:
                ai_sl, ai_tp = ai_decision.get("stop_loss"), ai_decision.get("take_profit")
                if ai_sl is not None:
                    planned_sl = max(planned_sl, ai_sl) if is_buy else min(planned_sl, ai_sl)
                if ai_tp is not None:
                    if is_buy and ai_tp > planned_tp:
                        planned_tp = ai_tp
                    elif not is_buy and ai_tp < planned_tp:
                        planned_tp = ai_tp

            changed = await asyncio.to_thread(modify_position_protection, ticket, planned_sl, planned_tp)
            if changed["changed"]:
                append_management_event(ticket, {"action": "PROTECTION_UPDATE", "stop_loss": changed["stop_loss"],
                    "take_profit": changed["take_profit"], "ai_action": ai_decision["action"],
                    "ai_confidence": ai_decision["confidence_score"], "reason": ai_decision["logic"]})
                log.info("[TRADE] [EXIT MANAGER] %s #%s SL=%s TP=%s", symbol, ticket, changed["stop_loss"], changed["take_profit"])
                bot_state["last_exit_logic"] = f"{symbol} #{ticket}: protection updated. SL {changed['stop_loss']}, TP {changed['take_profit']}. {ai_decision['logic']}"
            _record_exit_review(position, ai_decision["display_action"], ai_decision["display_confidence"],
                                ai_decision["display_logic"], planned_sl, planned_tp, ai_decision.get("reviewed_at"))
        except Exception as exc:
            bot_state["last_exit_logic"] = f"{symbol} #{ticket}: exit review/update failed; existing broker SL/TP remain in place. {exc}"
            previous_review = _exit_reviews_by_ticket.get(ticket)
            if previous_review and previous_review.get("confidence_score") is not None:
                _record_exit_review(position, previous_review["action"], previous_review["confidence_score"],
                                    f"{previous_review['logic']} Latest position check failed: {exc}")
            else:
                _record_exit_review(position, "ERROR", None, f"Exit review/update failed; existing broker protection remains in place. {exc}")
            log.warning("[TRADE] Exit management failed for %s #%s: %s", symbol, ticket, exc)


async def trading_loop() -> None:
    global _next_entry_scan, _next_position_check
    while True:
        now = time.monotonic()
        if bot_state["is_running"] and now >= _next_entry_scan:
            async with _engine_lock:
                for symbol in config.SYMBOLS:
                    if not bot_state["is_running"]:
                        break
                    try:
                        await _scan_symbol(symbol)
                    except Exception as exc:
                        bot_state["last_error"] = f"{symbol}: {exc}"
                        log.exception("[SYSTEM] Scan failed for %s", symbol)
                        if "retcode=10026" in str(exc) or "AutoTrading disabled by server" in str(exc):
                            bot_state["is_running"] = False
                            log.error("[SYSTEM] Engine paused: MT5 server has disabled algorithmic trading")
                            break
            _next_entry_scan = time.monotonic() + bot_state["interval"]
        # Position protection/reconciliation has its own cadence, independent of entry scans.
        if time.monotonic() >= _next_position_check:
            await _manage_open_positions()
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
