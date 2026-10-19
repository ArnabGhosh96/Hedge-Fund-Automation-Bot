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
