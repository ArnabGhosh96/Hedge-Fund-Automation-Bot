"""LLM-assisted post-trade loss analysis and persistent risk-rule learning."""
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

import config
from memory_store import load_trade_memory, reconcile_closed_trades, _atomic_save

log = logging.getLogger(__name__)
EMPTY_DOC = {"rules": [], "last_audit_at": None, "trades_analyzed": 0}


def load_rules_document() -> dict[str, Any]:
    try:
        with open(config.RULES_FILE, "r", encoding="utf-8") as stream:
            doc = json.load(stream)
        if not isinstance(doc, dict) or not isinstance(doc.get("rules"), list):
            return dict(EMPTY_DOC)
        return {**EMPTY_DOC, **doc}
    except (OSError, json.JSONDecodeError):
        return dict(EMPTY_DOC)


def save_rules_document(doc: dict[str, Any]) -> None:
    _atomic_save(config.RULES_FILE, doc)


def _parse_json(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    return json.loads(text)


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
    losses = [r for r in closed if float(r["realized_pnl"]) < 0]
    if not losses:
        doc.update(last_audit_at=now.isoformat(), trades_analyzed=len(closed))
        save_rules_document(doc)
        return {"status": "no_losses", "closed_trades": len(closed), "newly_reconciled": newly_reconciled}
    summary = [{k: r.get(k) for k in ("symbol", "side", "entry_price", "exit_price", "realized_pnl", "market_context")} for r in losses]
    if not config.DEEPSEEK_API_KEY:
        return {"status": "api_key_missing", "losses": len(losses), "newly_reconciled": newly_reconciled}
    system = '''You are the Chief Risk Officer and Quantitative Auditor. Analyze only the supplied realized losing trades. Find recurring low-volume breakdown/breakout failures, ATR overextension or volatility traps, and adverse cross-asset correlations. Return raw JSON only: {"rules":[{"affected_symbol":"SYMBOL or ALL","setup":"specific observable condition","confidence_reduction_points":15-30,"sample_size":integer,"evidence":"brief evidence grounded in input"}]}. Do not invent correlations or claim causality from weak evidence. Return an empty rules array if no repeatable pattern is supported.'''
    try:
        response = requests.post(f"{config.DEEPSEEK_API_BASE}/chat/completions", headers={"Authorization": f"Bearer {config.DEEPSEEK_API_KEY}"},
                                 json={"model": config.DEEPSEEK_MODEL, "temperature": 0.1, "messages": [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(summary, default=str, allow_nan=False)}]}, timeout=(5, 45))
        response.raise_for_status()
        raw_rules = _parse_json(response.json()["choices"][0]["message"]["content"])["rules"]
        if not isinstance(raw_rules, list):
            raise ValueError("Auditor rules field must be a list")
        valid = []
        for rule in raw_rules:
            if not isinstance(rule, dict):
                continue
            try:
                penalty = int(rule["confidence_reduction_points"])
                sample = int(rule["sample_size"])
                symbol = str(rule["affected_symbol"]).strip().upper()
                setup, evidence = str(rule["setup"]).strip(), str(rule["evidence"]).strip()
                if not symbol or not setup or penalty not in range(15, 31) or sample < 1 or not evidence:
                    continue
                valid.append({"affected_symbol": symbol, "setup": setup[:500], "confidence_reduction_points": penalty,
                              "sample_size": sample, "evidence": evidence[:1200], "active": True, "created_at": now.isoformat()})
            except (KeyError, ValueError, TypeError):
                continue
        # Keep historical rules and add only valid new findings.
        doc["rules"] = (doc.get("rules", []) + valid)[-200:]
        doc.update(last_audit_at=now.isoformat(), trades_analyzed=len(closed))
        save_rules_document(doc)
        log.info("[AUDITOR] analyzed=%s losses=%s new_rules=%s", len(closed), len(losses), len(valid))
        return {"status": "complete", "closed_trades": len(closed), "losses": len(losses), "rules_added": len(valid), "newly_reconciled": newly_reconciled, "last_audit_at": now.isoformat()}
    except (requests.RequestException, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        log.exception("[AUDITOR] Audit failed")
        return {"status": "error", "detail": str(exc), "losses": len(losses), "newly_reconciled": newly_reconciled}
