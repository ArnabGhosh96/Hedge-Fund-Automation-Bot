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
VOL_MA_PERIOD = 20
AUDIT_TRADE_THRESHOLD: int = max(3, int(os.getenv("AUDIT_TRADE_THRESHOLD", "10")))
AUDIT_COOLDOWN_HOURS: int = max(0, int(os.getenv("AUDIT_COOLDOWN_HOURS", "12")))
MEMORY_FILE = BASE_DIR / "memory.json"
RULES_FILE = BASE_DIR / "new_rules.json"
MAX_RISK_PERCENT = 5.0
EXIT_CONFIDENCE_THRESHOLD: int = max(0, min(100, int(os.getenv("EXIT_CONFIDENCE_THRESHOLD", "75"))))
EXIT_AI_REVIEW_SECONDS: int = max(30, int(os.getenv("EXIT_AI_REVIEW_SECONDS", "300")))
POSITION_MONITOR_INTERVAL_SECONDS: int = max(5, min(3600, int(os.getenv("POSITION_MONITOR_INTERVAL_SECONDS", "300"))))
TRAIL_ACTIVATE_ATR: float = max(0.1, float(os.getenv("TRAIL_ACTIVATE_ATR", "1.0")))
TRAIL_DISTANCE_ATR: float = max(0.1, float(os.getenv("TRAIL_DISTANCE_ATR", "1.5")))
