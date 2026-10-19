"""FastAPI dashboard and autonomous, fail-closed scan engine."""
import asyncio
import logging
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
from ai_brain import get_ai_decision, load_learned_rules
from auditor import load_rules_document, run_audit
from data_engine import fetch_correlated_asset_prices, fetch_multi_timeframe_data, initialize_mt5
from execution import close_position, execute_trade, get_open_positions
from memory_store import append_trade_memory, load_trade_memory, reconcile_closed_trades

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))
bot_state: dict[str, Any] = {"is_running": False, "interval": config.SCAN_INTERVAL_SECONDS, "risk_percent": config.DEFAULT_RISK_PERCENT,
    "equity": 0.0, "last_logic": "Waiting for scan", "last_confidence": 0, "last_signal": "HOLD", "last_entry_price": None,
    "last_stop_loss": None, "last_take_profit": None, "trade_history": [], "open_positions": [], "learned_rules": [], "last_audit_at": None,
    "last_symbol": None, "last_scan_at": None, "last_error": None, "reconciled_count": 0}
_engine_lock = asyncio.Lock()
_audit_lock = asyncio.Lock()


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


async def trading_loop() -> None:
    while True:
        if bot_state["is_running"]:
            async with _engine_lock:
                for symbol in config.SYMBOLS:
                    if not bot_state["is_running"]:
                        break
                    try:
                        await _scan_symbol(symbol)
                    except Exception as exc:
                        bot_state["last_error"] = f"{symbol}: {exc}"
                        log.exception("[SYSTEM] Scan failed for %s", symbol)
                try:
                    newly_closed = await asyncio.to_thread(reconcile_closed_trades)
                    bot_state["reconciled_count"] += newly_closed
                    if newly_closed:
                        await _audit(True)
                    await _refresh_state()
                except Exception as exc:
                    bot_state["last_error"] = f"Reconciliation: {exc}"
                    log.exception("[SYSTEM] Reconciliation failed")
        await asyncio.sleep(max(1, bot_state["interval"]))


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await asyncio.to_thread(initialize_mt5)
        await asyncio.to_thread(reconcile_closed_trades)
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
    await _refresh_state()
    return bot_state


@app.post("/api/control")
async def control(payload: ControlRequest):
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
    bot_state["is_running"] = action == "start"
    log.info("[SYSTEM] Engine %s interval=%ss risk=%s%%", action.upper(), bot_state["interval"], bot_state["risk_percent"])
    return {"ok": True, "is_running": bot_state["is_running"], "interval": bot_state["interval"], "risk_percent": bot_state["risk_percent"]}


@app.post("/api/close")
async def close(payload: CloseRequest):
    try:
        result = await asyncio.to_thread(close_position, payload.ticket)
        await _refresh_state()
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
