"""Durable JSON trade ledger and MT5 history reconciliation."""
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config

log = logging.getLogger(__name__)


def _read_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as stream:
            value = json.load(stream)
            return value if isinstance(value, type(default)) else default
    except (OSError, json.JSONDecodeError):
        return default


def _atomic_save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, default=str)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def load_trade_memory() -> list[dict[str, Any]]:
    return _read_json(config.MEMORY_FILE, [])


def realized_calibration(symbol: str) -> dict[str, Any]:
    """Closed-trade dossier for the AI prompt and the calibration gate.

    Uses the same closed-trade definition as the auditor and P/L ledger (status CLOSED with a
    numeric realized_pnl). Returns {"symbol": per-symbol stats, "confidence_bands": stats per
    reported-confidence bucket sharing ALL symbols' closed trades}. Empty/None stats when
    nothing has closed yet, so consumers can degrade gracefully.
    """
    closed = [r for r in load_trade_memory()
              if r.get("status") == "CLOSED" and isinstance(r.get("realized_pnl"), (int, float))]

    def _stats(rows: list[float]) -> dict[str, Any]:
        total = len(rows)
        wins = sum(1 for pnl in rows if pnl > 1e-9)
        return {"closed": total, "wins": wins, "win_rate": round(wins / total, 3) if total else None,
                "avg_pnl": round(sum(rows) / total, 2) if total else None, "losses": total - wins}

    symbols = _stats([float(r["realized_pnl"]) for r in closed if r.get("symbol") == symbol])
    bands = []
    for low, high, label in ((0, 50, "<50"), (50, 60, "50-59"), (60, 70, "60-69"), (70, 101, "70+")):
        rows = [float(r["realized_pnl"]) for r in closed
                if isinstance(r.get("confidence"), int) and not isinstance(r.get("confidence"), bool)
                and low <= r["confidence"] < high]
        bands.append({"band": label, **_stats(rows)})
    return {"symbol": symbols, "confidence_bands": bands}


_DECISIONS_APPENDED = 0


def _atomic_write_text(path: Path, text: str) -> None:
    fd, temp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def append_decision_memory(record: dict[str, Any]) -> None:
    """Append one AI decision as a JSONL line; never raises so the scan keeps flowing."""
    global _DECISIONS_APPENDED
    try:
        line = json.dumps(record, ensure_ascii=False, default=str, allow_nan=False)
        with open(config.DECISIONS_FILE, "a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        _DECISIONS_APPENDED += 1
        if _DECISIONS_APPENDED >= config.DECISION_PRUNE_EVERY:
            _DECISIONS_APPENDED = 0
            prune_decision_store()
    except (OSError, ValueError) as exc:  # OSError = disk issues; ValueError = non-JSON-serializable values
        log.warning("[MEMORY] decision not persisted: %s", exc)


def prune_decision_store() -> None:
    """Keep only the newest DECISION_STORE_LIMIT records to bound journal growth."""
    limit = config.DECISION_STORE_LIMIT
    try:
        with open(config.DECISIONS_FILE, "r", encoding="utf-8") as stream:
            lines = stream.readlines()
        if len(lines) <= limit:
            return
        _atomic_write_text(config.DECISIONS_FILE, "".join(lines[-limit:]))
        log.info("[MEMORY] decision store pruned to newest %s records", limit)
    except OSError as exc:
        log.warning("[MEMORY] decision store prune failed: %s", exc)


def load_recent_decisions(symbol: str, limit: int) -> list[dict[str, Any]]:
    """Read the newest `limit` decisions for one symbol from the tail of the JSONL store.

    Only the cheap decision-level fields come back (the heavy candle arrays stay on disk),
    newest last, ready to feed back into the model's prompt for continuity.
    """
    if limit <= 0 or not config.DECISIONS_FILE.exists():
        return []
    try:
        size = config.DECISIONS_FILE.stat().st_size
        window = min(size, 1_048_576)  # 1MB tail is far more than an hour of records
        with open(config.DECISIONS_FILE, "rb") as stream:
            if window < size:
                stream.seek(size - window)
                stream.readline()  # drop the partial line at the window boundary
            raw = stream.read()
        records: list[dict[str, Any]] = []
        for line in reversed(raw.decode("utf-8", "replace").splitlines()):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(record.get("symbol", "")).upper() != symbol.upper():
                continue
            records.append({"at": record.get("at"), "signal": record.get("signal"),
                            "confidence": record.get("confidence"), "stop_loss": record.get("stop_loss"),
                            "take_profit": record.get("take_profit"), "mid_price": record.get("mid_price"),
                            "logic": str(record.get("logic") or "")[:220]})
            if len(records) >= limit:
                break
        records.reverse()
        return records
    except OSError as exc:
        log.warning("[MEMORY] decision history unavailable: %s", exc)
        return []


def append_trade_memory(record: dict[str, Any]) -> None:
    records = load_trade_memory()
    records.append({"status": "CONFIRMED", "created_at": datetime.now(timezone.utc).isoformat(), **record})
    _atomic_save(config.MEMORY_FILE, records)


def append_management_event(position_id: int, event: dict[str, Any]) -> None:
    """Attach exit-management decisions and protection changes to the original trade record."""
    records = load_trade_memory()
    target = None
    for record in reversed(records):
        ids = {int(record.get(key) or 0) for key in ("position_id", "ticket", "order")}
        if int(position_id) in ids:
            target = record
            break
    if target is None:
        log.warning("[TRADE] No local ledger record found for position %s exit event", position_id)
        return
    target.setdefault("management_events", []).append({"at": datetime.now(timezone.utc).isoformat(), **event})
    if event.get("stop_loss") is not None:
        target["stop_loss"] = event["stop_loss"]
    if event.get("take_profit") is not None:
        target["take_profit"] = event["take_profit"]
    _atomic_save(config.MEMORY_FILE, records)


def reconcile_closed_trades() -> int:
    try:
        import MetaTrader5 as mt5
    except ImportError:
        log.warning("[SYSTEM] MT5 unavailable; trade reconciliation skipped")
        return 0
    records = load_trade_memory()
    pending = [r for r in records if r.get("status") == "CONFIRMED"]
    if not pending:
        return 0
    now = datetime.now(timezone.utc)
    history = mt5.history_deals_get(datetime(2000, 1, 1, tzinfo=timezone.utc), now)
    if history is None:
        log.warning("[SYSTEM] MT5 deal history unavailable: %s", mt5.last_error())
        return 0
    by_position: dict[int, list[Any]] = {}
    opening_deals: dict[int, Any] = {}
    for deal in history:
        position_id = int(getattr(deal, "position_id", 0) or 0)
        if position_id:
            by_position.setdefault(position_id, []).append(deal)
        opening_deals[int(getattr(deal, "ticket", 0) or 0)] = deal

    # A deal's order ticket is not guaranteed to be its position identifier.
    # Resolve old ledger records through their opening deal when possible.
    live_positions = mt5.positions_get()
    if live_positions is None:
        log.warning("[SYSTEM] MT5 positions unavailable during reconciliation: %s", mt5.last_error())
        return 0
    live_ids = {int(getattr(p, "identifier", 0) or getattr(p, "ticket", 0) or 0) for p in live_positions}
    changed = 0
    for record in pending:
        position_id = int(record.get("position_id") or 0)
        entry = opening_deals.get(int(record.get("deal") or 0))
        if entry is None:
            order_id = int(record.get("order") or record.get("ticket") or 0)
            entry = next((d for d in history if int(getattr(d, "order", 0) or 0) == order_id
                          and getattr(d, "entry", None) == getattr(mt5, "DEAL_ENTRY_IN", 0)), None)
        resolved_id = int(getattr(entry, "position_id", 0) or 0) if entry is not None else 0
        if resolved_id:
            position_id = resolved_id
        if not position_id:
            continue
        # MT5 can return an incomplete account-wide history slice (for example,
        # when terminal history is limited) while its position-specific query
        # still returns the complete lifecycle. Prefer the latter.
        position_history = mt5.history_deals_get(position=position_id)
        if position_history is None:
            log.warning("[SYSTEM] MT5 history query failed for position %s: %s", position_id, mt5.last_error())
            candidates = by_position.get(position_id, [])
        else:
            candidates = list(position_history)
        exit_entries = {getattr(mt5, "DEAL_ENTRY_OUT", 1), getattr(mt5, "DEAL_ENTRY_OUT_BY", 3)}
        exits = [d for d in candidates if getattr(d, "entry", None) in exit_entries]
        if not exits:
            continue
        # A partial close is not a closed position. Wait until MT5 no longer
        # reports that position identifier as open, then account for all exits.
        if position_id in live_ids:
            continue
        exits.sort(key=lambda d: int(getattr(d, "time", 0)))
        deal = exits[-1]
        pnl = sum(float(getattr(d, "profit", 0.0) or 0.0) + float(getattr(d, "commission", 0.0) or 0.0)
                  + float(getattr(d, "swap", 0.0) or 0.0) + float(getattr(d, "fee", 0.0) or 0.0) for d in exits)
        record.update({"status": "CLOSED", "position_id": position_id,
                       "exit_deal": int(deal.ticket), "exit_deals": [int(d.ticket) for d in exits],
                       "exit_price": float(deal.price),
                       "exit_time": datetime.fromtimestamp(int(deal.time), timezone.utc).isoformat(),
                       "realized_pnl": pnl, "outcome": "WIN" if pnl > 1e-9 else "LOSS" if pnl < -1e-9 else "BREAKEVEN"})
        changed += 1
    if changed:
        _atomic_save(config.MEMORY_FILE, records)
        log.info("[TRADE] Reconciled %s closed trade(s)", changed)
    return changed
