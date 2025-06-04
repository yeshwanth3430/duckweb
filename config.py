import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_DATA_DIR = BASE_DIR / "sample_data"

# Database configuration
DB_PATH = os.getenv("DB_PATH", str(DATA_DIR / "market_data.duckdb"))
SAMPLE_DB_PATH = str(SAMPLE_DATA_DIR / "sample_market_data.duckdb")

# Streamlit configuration
STREAMLIT_CONFIG = {
    "theme": {
        "primaryColor": "#1E88E5",
        "backgroundColor": "#FFFFFF",
        "secondaryBackgroundColor": "#F0F2F6",
        "textColor": "#262730",
        "font": "sans serif"
    }
}

# Analysis configuration
DEFAULT_TIMEFRAMES = ["1min", "hourly", "daily"]
DEFAULT_INDICES = ["nifty", "banknifty", "sensex"]

# Technical analysis parameters
TECHNICAL_PARAMS = {
    "sma_periods": [20, 50, 200],
    "ema_periods": [20, 50, 200],
    "macd_params": {
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9
    },
    "supertrend_params": {
        "period": 10,
        "multiplier": 3
    }
}

# Create necessary directories
DATA_DIR.mkdir(exist_ok=True)
SAMPLE_DATA_DIR.mkdir(exist_ok=True) 