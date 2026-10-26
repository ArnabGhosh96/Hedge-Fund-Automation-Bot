"""LLM-assisted post-trade loss analysis with ONE standing risk rule per instrument.

Each audit aggregates the durable ledger per symbol (closed-trade counts, win/loss
split, realized P/L) and asks the auditor model to MODIFY each instrument's standing
rule in light of that evidence — merging instead of appending, so the decision prompt
never sees multiple rules for the same instrument (plus one optional global ALL rule).
"""
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

import config
from memory_store import load_trade_memory, reconcile_closed_trades, _atomic_save

log = logging.getLogger(__name__)
EMPTY_DOC = {"rules": [], "last_audit_at": None, "trades_analyzed": 0}
SUMMARY_KEYS = ("symbol", "side", "entry_price", "exit_price", "realized_pnl", "market_context")


def load_rules_document() -> dict[str, Any]:
    try:
        with open(config.RULES_FILE, "r", encoding="utf-8") as stream:
            doc = json.load(stream)
        if not isinstance(doc, dict) or not isinstance(doc.get("rules"), list):
            return dict(EMPTY_DOC)
        return {**EMPTY_DOC, **doc}
    except (OSError, json.JSONDecodeError):
        return dict(EMPTY_DOC)


def _rule_age(rule: dict[str, Any]) -> Any:
    """Best available wall-clock anchor for 'how fresh is this rule' (None if missing)."""
    for field in ("updated_at", "created_at"):
        value = str(rule.get(field) or "")
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            continue
    return None


def collapse_rules(rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fold legacy duplicate rules into at most one per affected_symbol.

    Audits used to append a rule per finding, so instruments could accumulate many
    near-identical rules. Survivors are chosen by timestamp freshness (updated_at,
    then created_at) — not list position, because merges can replace earlier slots —
    so the newest standing rule for each instrument keeps its content. Record counts
    (sample_size) stay as-is; the ledger refresh in run_audit re-truths them anyway.
    """
    by_key: dict[str, dict[str, Any]] = {}
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        key = str(rule.get("affected_symbol", "")).strip().upper()
        if not key:
            continue
        current = by_key.get(key)
        if current is None:
            by_key[key] = dict(rule)
            continue
        rule_age, current_age = _rule_age(rule), _rule_age(current)
        if rule_age is None and current_age is None:
            by_key[key] = dict(rule)  # no timestamps anywhere: later position wins
        elif rule_age is not None and (current_age is None or rule_age >= current_age):
            by_key[key] = dict(rule)
        # else: the already-stored (newer) record stays
    return list(by_key.values())


def save_rules_document(doc: dict[str, Any]) -> None:
    doc["rules"] = collapse_rules(doc.get("rules", []))[-200:]
    _atomic_save(config.RULES_FILE, doc)


def closed_trade_stats(closed: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Aggregate closed trades per symbol plus an ALL bucket, straight from the ledger."""
    by_key: dict[str, dict[str, Any]] = {}
    for trade in closed:
        key = str(trade.get("symbol", "")).strip().upper()
        if not key:
            continue
        try:
            pnl = float(trade.get("realized_pnl") or 0.0)
        except (TypeError, ValueError):
            continue
        summary = {k: trade.get(k) for k in SUMMARY_KEYS}
        for bucket in (key, "ALL"):
            stats = by_key.setdefault(bucket, {"closed": 0, "wins": 0, "losses": 0, "total_pnl": 0.0, "losing": []})
            stats["closed"] += 1
            stats["wins"] += 1 if pnl > 1e-9 else 0
            stats["losses"] += 1 if pnl < -1e-9 else 0
            stats["total_pnl"] += pnl
            if pnl < 0:
                stats["losing"].append(summary)
    return by_key


def _parse_json(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    return json.loads(text)


def _refresh_rule_stats(doc: dict[str, Any], stats_by_key: dict[str, dict[str, Any]]) -> None:
    """Update factual counters on existing rules; content itself stays untouched."""
    for rule in doc.get("rules", []):
        stats = stats_by_key.get(str(rule.get("affected_symbol", "")).upper())
        if not stats:
            continue
        rule["sample_size"] = stats["closed"]
        rule["wins"] = stats["wins"]
        rule["losses"] = stats["losses"]
        rule["total_realized_pnl"] = round(stats["total_pnl"], 2)


def _upsert_rule(rules: list[dict[str, Any]], merged: dict[str, Any]) -> None:
    for i, old in enumerate(rules):
        if str(old.get("affected_symbol", "")).upper() == merged["affected_symbol"]:
            rules[i] = merged
            return
    rules.append(merged)


def run_audit(force: bool = False) -> dict[str, Any]:
    newly_reconciled = reconcile_closed_trades()
    doc = load_rules_document()
    closed = [r for r in load_trade_memory() if r.get("status") == "CLOSED" and isinstance(r.get("realized_pnl"), (int, float))]
    closed = closed[-50:]
    if len(closed) < 3:
        return {"status": "insufficient_trades", "closed_trades": len(closed), "newly_reconciled": newly_reconciled}
    unanalyzed = len(closed)
    if doc.get("last_audit_at"):
        try:
            last_audit = datetime.fromisoformat(doc["last_audit_at"].replace("Z", "+00:00"))
            unanalyzed = sum(1 for trade in closed if trade.get("exit_time") and
                             datetime.fromisoformat(trade["exit_time"].replace("Z", "+00:00")) > last_audit)
        except (ValueError, TypeError):
            unanalyzed = len(closed)
    if not force and unanalyzed < config.AUDIT_TRADE_THRESHOLD:
        return {"status": "threshold_not_met", "closed_trades": len(closed), "unanalyzed_trades": unanalyzed,
                "required": config.AUDIT_TRADE_THRESHOLD, "newly_reconciled": newly_reconciled}
    now = datetime.now(timezone.utc)
    last = doc.get("last_audit_at")
    if not force and last:
        try:
            if now - datetime.fromisoformat(last.replace("Z", "+00:00")) < timedelta(hours=config.AUDIT_COOLDOWN_HOURS):
                return {"status": "cooldown", "last_audit_at": last, "closed_trades": len(closed)}
        except (ValueError, TypeError):
            pass
    stats_by_key = closed_trade_stats(closed)
    focus = {key: stats for key, stats in sorted(stats_by_key.items()) if stats["losses"]}
    losing_trade_count = sum(stats["losses"] for stats in focus.values())
    _refresh_rule_stats(doc, stats_by_key)  # counters stay truthful even when the model is not consulted
    if not focus:
        doc.update(last_audit_at=now.isoformat(), trades_analyzed=len(closed))
        save_rules_document(doc)
        return {"status": "no_losses", "closed_trades": len(closed), "rules_total": len(doc["rules"]),
                "newly_reconciled": newly_reconciled}
    if not config.DEEPSEEK_API_KEY:
        return {"status": "api_key_missing", "losses": losing_trade_count, "newly_reconciled": newly_reconciled}
    existing = {str(r.get("affected_symbol", "")).upper(): r for r in collapse_rules(doc.get("rules", []))}
    payload = {}
    for key, stats in focus.items():
        rule = existing.get(key)
        payload[key] = {"closed_trades": stats["closed"], "wins": stats["wins"], "losses": stats["losses"],
                        "total_realized_pnl": round(stats["total_pnl"], 2),
                        "losing_trades": stats["losing"][-10:],
                        "current_rule": None if not rule else
                        {k: rule.get(k) for k in ("setup", "confidence_reduction_points", "sample_size", "evidence", "updated_at")}}
    system = """You are the Chief Risk Officer and Quantitative Auditor maintaining ONE standing risk rule per instrument — never per-trade duplicates.
For each instrument in the payload you receive its closed-trade record (trades, wins, losses, realized P/L, recent losing-trade details) and its current standing rule if one exists.
Decide how that rule should change in light of the new evidence: keep clauses that still describe an observed failure pattern, drop or rewrite clauses the record no longer supports, and state the setup as one specific observable condition the engine can recognize. Write evidence cumulatively so it reflects all trades so far. Adjust the penalty in light of the win rate as well — entrenched repeat losses justify the high end, occasional winners argue for the low end.
Return raw JSON only: {"rules":[{"affected_symbol":"SYMBOL or ALL","setup":"specific observable condition","confidence_reduction_points":15-30,"evidence":"cumulative evidence grounded in the record"}]}.
Only include instruments worth encoding (or a single ALL rule when one failure pattern repeats across instruments). Do not invent correlations or claim causality from weak evidence. An empty rules array is a valid answer."""
    try:
        response = requests.post(f"{config.DEEPSEEK_API_BASE}/chat/completions", headers={"Authorization": f"Bearer {config.DEEPSEEK_API_KEY}"},
                                 json={"model": config.DEEPSEEK_MODEL, "temperature": 0.1,
                                       "messages": [{"role": "system", "content": system},
                                                    {"role": "user", "content": json.dumps(payload, default=str, allow_nan=False)}]},
                                 timeout=(5, 45))
        response.raise_for_status()
        raw_rules = _parse_json(response.json()["choices"][0]["message"]["content"])["rules"]
        if not isinstance(raw_rules, list):
            raise ValueError("Auditor rules field must be a list")
        merged_keys: list[str] = []
        for rule in raw_rules:
            if not isinstance(rule, dict):
                continue
            try:
                key = str(rule["affected_symbol"]).strip().upper()
                penalty = int(rule["confidence_reduction_points"])
                setup, evidence = str(rule["setup"]).strip(), str(rule["evidence"]).strip()
                if not key or not setup or penalty not in range(15, 31) or not evidence:
                    continue
                stats = focus.get(key)
                if not stats:  # refuse symbols with no closed-trade evidence in the ledger
                    continue
                previous = existing.get(key) or {}
                _upsert_rule(doc["rules"], {"affected_symbol": key, "setup": setup[:500],
                                            "confidence_reduction_points": penalty, "sample_size": stats["closed"],
                                            "evidence": evidence[:1200], "active": True,
                                            "created_at": previous.get("created_at") or now.isoformat(),
                                            "updated_at": now.isoformat(), "wins": stats["wins"], "losses": stats["losses"],
                                            "total_realized_pnl": round(stats["total_pnl"], 2)})
                merged_keys.append(key)
            except (KeyError, TypeError, ValueError):
                continue
        _refresh_rule_stats(doc, stats_by_key)
        doc.update(last_audit_at=now.isoformat(), trades_analyzed=len(closed))
        save_rules_document(doc)
        log.info("[AUDITOR] analyzed=%s losing_symbols=%s rule_upserts=%s rules_total=%s",
                 len(closed), len(focus), len(merged_keys), len(doc["rules"]))
        return {"status": "complete", "closed_trades": len(closed),
                "losses": sum(stats["losses"] for stats in focus.values()),
                "rules_added": len(merged_keys), "rules_total": len(doc["rules"]),
                "newly_reconciled": newly_reconciled, "last_audit_at": now.isoformat()}
    except (requests.RequestException, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        log.exception("[AUDITOR] Audit failed")
        return {"status": "error", "detail": str(exc), "losses": losing_trade_count, "newly_reconciled": newly_reconciled}
