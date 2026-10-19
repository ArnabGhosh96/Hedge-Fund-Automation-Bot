"""Durable JSON trade ledger and MT5 history reconciliation."""
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
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


def append_trade_memory(record: dict[str, Any]) -> None:
    records = load_trade_memory()
    records.append({"status": "CONFIRMED", "created_at": datetime.now(timezone.utc).isoformat(), **record})
    _atomic_save(config.MEMORY_FILE, records)


def reconcile_closed_trades() -> int:
    try:
        import MetaTrader5 as mt5
    except ImportError:
        log.warning("[SYSTEM] MT5 unavailable; trade reconciliation skipped")
        return 0
    records = load_trade_memory()
    pending = [r for r in records if r.get("status") == "CONFIRMED" and not r.get("exit_deal")]
    if not pending:
        return 0
    now = datetime.now(timezone.utc)
    history = mt5.history_deals_get(datetime(2000, 1, 1, tzinfo=timezone.utc), now)
    if history is None:
        log.warning("[SYSTEM] MT5 deal history unavailable: %s", mt5.last_error())
        return 0
    by_position: dict[int, list[Any]] = {}
    for deal in history:
        by_position.setdefault(int(getattr(deal, "position_id", 0) or 0), []).append(deal)
    changed = 0
    for record in pending:
        position_id = int(record.get("position_id") or record.get("order") or record.get("ticket") or 0)
        candidates = by_position.get(position_id, [])
        exits = [d for d in candidates if getattr(d, "entry", None) == getattr(mt5, "DEAL_ENTRY_OUT", 1)]
        if not exits:
            continue
        deal = max(exits, key=lambda d: int(getattr(d, "time", 0)))
        pnl = float(getattr(deal, "profit", 0.0) or 0.0) + float(getattr(deal, "commission", 0.0) or 0.0) + float(getattr(deal, "swap", 0.0) or 0.0) + float(getattr(deal, "fee", 0.0) or 0.0)
        record.update({"status": "CLOSED", "exit_deal": int(deal.ticket), "exit_price": float(deal.price),
                       "exit_time": datetime.fromtimestamp(int(deal.time), timezone.utc).isoformat(),
                       "realized_pnl": pnl, "outcome": "WIN" if pnl > 1e-9 else "LOSS" if pnl < -1e-9 else "BREAKEVEN"})
        changed += 1
    if changed:
        _atomic_save(config.MEMORY_FILE, records)
        log.info("[TRADE] Reconciled %s closed trade(s)", changed)
    return changed
