"""Environment-backed application configuration."""
from pathlib import Path
import os

from dotenv import load_dotenv

try:
    import MetaTrader5 as mt5
except ImportError:  # Allows dashboard/config tooling to start without the Windows MT5 package.
    class _MT5Constants:
        TIMEFRAME_H1 = 16385
        TIMEFRAME_D1 = 16408

    mt5 = _MT5Constants()  # type: ignore[assignment]

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_API_BASE: str = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com").rstrip("/")
DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()
MT5_LOGIN: int | None = int(os.environ["MT5_LOGIN"]) if os.getenv("MT5_LOGIN", "").strip() else None
MT5_PASSWORD: str = os.getenv("MT5_PASSWORD", "")
MT5_SERVER: str = os.getenv("MT5_SERVER", "")
MT5_PATH: str | None = os.getenv("MT5_PATH", "").strip() or None
DEFAULT_RISK_PERCENT: float = float(os.getenv("DEFAULT_RISK_PERCENT", "1.0"))
SCAN_INTERVAL_SECONDS: int = max(5, int(os.getenv("SCAN_INTERVAL_SECONDS", "30")))
CONFIDENCE_THRESHOLD: int = max(0, min(100, int(os.getenv("CONFIDENCE_THRESHOLD", "65"))))
SYMBOLS: list[str] = [s.strip() for s in os.getenv(
    "SYMBOLS", "EURUSDm,GBPUSDm,USDJPYm,USDCHFm,USDCADm,AUDUSDm,NZDUSDm,BTCUSDm,XAUUSDm"
).split(",") if s.strip()]
TIMEFRAMES = {"H1": mt5.TIMEFRAME_H1, "D1": mt5.TIMEFRAME_D1}
EMA_PERIOD = 200
RSI_PERIOD = 14
ATR_PERIOD = 14
ADX_PERIOD = 14
VOL_MA_PERIOD = 20
AUDIT_TRADE_THRESHOLD: int = max(3, int(os.getenv("AUDIT_TRADE_THRESHOLD", "10")))
AUDIT_COOLDOWN_HOURS: int = max(0, int(os.getenv("AUDIT_COOLDOWN_HOURS", "12")))
MEMORY_FILE = BASE_DIR / "memory.json"
RULES_FILE = BASE_DIR / "new_rules.json"
# Append-only AI decision journal; consumed for prompt continuity and later reviews/audits.
DECISIONS_FILE = BASE_DIR / "ai_decisions.jsonl"
MAX_RISK_PERCENT = 5.0
POSITION_MONITOR_INTERVAL_SECONDS: int = max(5, min(3600, int(os.getenv("POSITION_MONITOR_INTERVAL_SECONDS", "300"))))

# LLM request tuning. JSON mode requests strict JSON output (auto-falls back if unsupported).
AI_JSON_MODE: bool = os.getenv("AI_JSON_MODE", "true").strip().lower() not in {"0", "false", "no", "off"}
AI_MAX_ATTEMPTS: int = max(1, min(4, int(os.getenv("AI_MAX_ATTEMPTS", "2"))))
AI_RETRY_BACKOFF_SECONDS: float = max(0.5, min(30.0, float(os.getenv("AI_RETRY_BACKOFF_SECONDS", "1.0"))))
# Streaming avoids whole-response read timeouts: the read timeout then caps silence, not latency.
AI_STREAM: bool = os.getenv("AI_STREAM", "true").strip().lower() not in {"0", "false", "no", "off"}
AI_CONNECT_TIMEOUT_SECONDS: float = max(1.0, min(30.0, float(os.getenv("AI_CONNECT_TIMEOUT_SECONDS", "5"))))
AI_READ_TIMEOUT_SECONDS: float = max(5.0, min(300.0, float(os.getenv("AI_READ_TIMEOUT_SECONDS", "60"))))

# Deterministic pre-filters, evaluated before any LLM call so a skipped symbol costs zero API budget.
SKIP_SPREAD_ATR_RATIO: float = max(0.0, float(os.getenv("SKIP_SPREAD_ATR_RATIO", "0.15")))
SKIP_MIN_RELATIVE_VOLUME: float = max(0.0, float(os.getenv("SKIP_MIN_RELATIVE_VOLUME", "0")))
SKIP_CLOSED_MARKETS: bool = os.getenv("SKIP_CLOSED_MARKETS", "true").strip().lower() not in {"0", "false", "no", "off"}
ALWAYS_OPEN_SYMBOLS: frozenset[str] = frozenset(
    s.strip().upper() for s in os.getenv("ALWAYS_OPEN_SYMBOLS", "BTCUSD,ETHUSD").split(",") if s.strip())

# Prompt data sizing: how many candles go to the AI (see .env comments for the trade-off).
RECENT_H1_BARS: int = max(0, min(125, int(os.getenv("RECENT_H1_BARS", "24"))))
RECENT_DAILY_BARS: int = max(0, min(60, int(os.getenv("RECENT_DAILY_BARS", "10"))))

# Daily circuit breaker: stop the engine when equity falls by this percent from the
# engine-start baseline on the current UTC day (0 disables).
DAILY_LOSS_STOP_PERCENT: float = max(0.0, min(50.0, float(os.getenv("DAILY_LOSS_STOP_PERCENT", "3"))))

# AI decision journal sizing (see .env comments for growth trade-offs).
DECISION_STORE_LIMIT: int = max(500, min(20000, int(os.getenv("DECISION_STORE_LIMIT", "3000"))))
DECISION_PRUNE_EVERY: int = max(50, min(5000, int(os.getenv("DECISION_PRUNE_EVERY", "250"))))
RECENT_DECISIONS_IN_PROMPT: int = max(0, min(6, int(os.getenv("RECENT_DECISIONS_IN_PROMPT", "3"))))

# Reverse-execution mode: model BUY executes as SELL, model SELL executes as BUY (HOLD unchanged).
INVERT_DECISIONS: bool = os.getenv("INVERT_DECISIONS", "false").strip().lower() in ("1", "true", "yes", "on")

# ---- Risk/reward shaping (executed side) -----------------------------------------------
# Enforced reward leg: TP = entry +/- MULTIPLE x shaped stop distance. 0 keeps model targets.
REWARD_RISK_MULTIPLE: float = max(0.0, min(10.0, float(os.getenv("REWARD_RISK_MULTIPLE", "2.0"))))
# ATR-multiple band the stop distance may not escape (snap, not reject; 0 disables an edge).
SL_MIN_ATR: float = max(0.0, min(10.0, float(os.getenv("SL_MIN_ATR", "1.0"))))
SL_MAX_ATR: float = max(0.0, min(10.0, float(os.getenv("SL_MAX_ATR", "1.5"))))
# Breakeven once price moves TRIGGER x remaining-risk in favor; then trail TRAIL x H1 ATR.
BREAKEVEN_TRIGGER_R: float = max(0.0, min(10.0, float(os.getenv("BREAKEVEN_TRIGGER_R", "1.0"))))
TRAIL_ATR_MULT: float = max(0.0, min(10.0, float(os.getenv("TRAIL_ATR_MULT", "1.0"))))
# Minimum stop improvement before sending an SLTP modify (prevents order-spam each pass).
TRAIL_MIN_STEP_ATR: float = max(0.0, min(2.0, float(os.getenv("TRAIL_MIN_STEP_ATR", "0.25"))))
# TP-less trailing mode: orders carry no take profit; the ATR trail is the only positive exit
# (winners unbounded, losers still capped at -1R). Overrides REWARD_RISK_MULTIPLE at execution
# time; the model's own TP is still journaled and shown as the model's intent.
TRAIL_ONLY_MODE: bool = os.getenv("TRAIL_ONLY_MODE", "false").strip().lower() in ("1", "true", "yes", "on")
# Magic number that brands this engine's orders; exit management only touches these.
BOT_MAGIC: int = int(os.getenv("BOT_MAGIC", "260019"))

# ---- Calibration loop: realized performance feeds back into decisions -------------------
# Each fresh decision carries the ledger's closed-trade dossier (per-symbol record plus the
# global mapping of confidence buckets to realized win-rates). A symbol whose closed win-rate
# sits below CALIBRATION_MIN_WINRATE once CALIBRATION_MIN_TRADES have closed must additionally
# clear CONFIDENCE_THRESHOLD + CALIBRATION_CONFIDENCE_PENALTY (engine-side gate; set 0 to
# disable the penalty while keeping the prompt dossier).
CALIBRATION_IN_PROMPT: bool = os.getenv("CALIBRATION_IN_PROMPT", "true").strip().lower() in ("1", "true", "yes", "on")
CALIBRATION_MIN_TRADES: int = max(1, int(os.getenv("CALIBRATION_MIN_TRADES", "5")))
CALIBRATION_MIN_WINRATE: float = max(0.0, min(1.0, float(os.getenv("CALIBRATION_MIN_WINRATE", "0.25"))))
CALIBRATION_CONFIDENCE_PENALTY: int = max(0, min(50, int(os.getenv("CALIBRATION_CONFIDENCE_PENALTY", "15"))))

# ---- Deterministic trade context added to every decision --------------------------------
KEY_LEVELS_IN_PROMPT: bool = os.getenv("KEY_LEVELS_IN_PROMPT", "true").strip().lower() in ("1", "true", "yes", "on")
SESSION_CONTEXT_IN_PROMPT: bool = os.getenv("SESSION_CONTEXT_IN_PROMPT", "true").strip().lower() in ("1", "true", "yes", "on")
# Swing-pivot width in bars (a pivot high/low must stand out on this many bars per side).
KEY_LEVEL_PIVOT_BARS: int = max(2, min(20, int(os.getenv("KEY_LEVEL_PIVOT_BARS", "3"))))
# At most this many resistance/support clusters are shown per side of the live price.
KEY_LEVEL_MAX_PER_SIDE: int = max(1, min(10, int(os.getenv("KEY_LEVEL_MAX_PER_SIDE", "4"))))
