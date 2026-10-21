"""DeepSeek decision layer with learned risk penalties and strict validation."""
import json
import logging
import math
from typing import Any

import requests

import config

log = logging.getLogger(__name__)


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


def get_ai_decision(market_data: dict[str, Any], symbol: str) -> dict[str, Any]:
    if not config.DEEPSEEK_API_KEY:
        return _fallback_hold(market_data, "DeepSeek API key is not configured; fail-closed HOLD.")
    rules = load_learned_rules(symbol)
    prompt_data = {"symbol": symbol, "bid": market_data["bid"], "ask": market_data["ask"], "spread": market_data["spread"],
                   "H1": market_data["h1_data"], "Daily": market_data["daily_data"], "active_rules": rules}
    system = '''You are a strict risk-managed institutional hedge-fund quantitative analyst. Return raw JSON only, exactly shaped as {"signal":"BUY"|"SELL"|"HOLD","confidence_score":0-100 integer,"stop_loss":number,"take_profit":number,"logic":"concise explanation including indicators and learned-rule deductions"}. Never invent data. Prefer HOLD when evidence is ambiguous, spread is excessive, indicators conflict, or required values are unavailable. Apply the specified confidence_reduction_points whenever a learned setup condition matches, and explain the deduction. Stops and targets must be geometrically valid around the live mid price.'''
    try:
        response = requests.post(f"{config.DEEPSEEK_API_BASE}/chat/completions", headers={"Authorization": f"Bearer {config.DEEPSEEK_API_KEY}"},
                                 json={"model": config.DEEPSEEK_MODEL, "temperature": 0.1, "messages": [{"role": "system", "content": system}, {"role": "user", "content": "Analyze this snapshot and active risk rules: " + json.dumps(prompt_data, allow_nan=False)}]}, timeout=(5, 30))
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()
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
        if confidence < config.CONFIDENCE_THRESHOLD:
            signal = "HOLD"
        result.update(signal=signal, confidence_score=confidence, stop_loss=sl, take_profit=tp, logic=str(result.get("logic", ""))[:2000])
        return result
    except (requests.RequestException, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        log.exception("[AI] DeepSeek decision failed for %s", symbol)
        return _fallback_hold(market_data, f"AI decision unavailable: {exc}")
