"""
Configuration management for the trading bot
"""
import os
from typing import Dict, List, Optional
from pydantic import BaseSettings, Field
from enum import Enum

class TradingMode(str, Enum):
    PAPER = "paper"
    LIVE = "live"

class RiskLevel(str, Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"

class Settings(BaseSettings):
    # Trading Configuration
    trading_mode: TradingMode = TradingMode.PAPER
    risk_level: RiskLevel = RiskLevel.MODERATE
    max_positions: int = 10
    max_daily_loss: float = 0.05  # 5% max daily loss
    max_position_size: float = 0.1  # 10% max position size
    
    # Exchange Configuration
    exchange_name: str = "binance"
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    sandbox: bool = True
    
    # Database Configuration
    database_url: str = "postgresql://trading_enhanced:enhanced_password@localhost:5433/trading_bot_enhanced"
    redis_url: str = "redis://localhost:6380/1"
    
    # Web Dashboard
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    # Security
    secret_key: str = "your-secret-key-change-this"
    access_token_expire_minutes: int = 30
    
    # ML Configuration
    ml_enabled: bool = True
    model_retrain_interval: int = 24  # hours
    feature_window: int = 100  # candles for feature engineering
    
    # Risk Management
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.04  # 4% take profit
    max_leverage: float = 3.0
    min_volume_24h: float = 1000000  # Minimum 24h volume
    
    # Trading Pairs
    trading_pairs: List[str] = [
        "BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "SOL/USDT",
        "XRP/USDT", "DOT/USDT", "AVAX/USDT", "MATIC/USDT", "LINK/USDT"
    ]
    
    # Timeframes
    timeframes: List[str] = ["1m", "5m", "15m", "1h", "4h", "1d"]
    primary_timeframe: str = "15m"
    
    # Monitoring
    log_level: str = "INFO"
    enable_metrics: bool = True
    sentry_dsn: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()

# Risk management parameters based on risk level
RISK_PARAMETERS = {
    RiskLevel.CONSERVATIVE: {
        "max_position_size": 0.05,
        "max_leverage": 1.5,
        "stop_loss_pct": 0.015,
        "take_profit_pct": 0.03,
        "max_daily_loss": 0.02
    },
    RiskLevel.MODERATE: {
        "max_position_size": 0.1,
        "max_leverage": 3.0,
        "stop_loss_pct": 0.02,
        "take_profit_pct": 0.04,
        "max_daily_loss": 0.05
    },
    RiskLevel.AGGRESSIVE: {
        "max_position_size": 0.2,
        "max_leverage": 5.0,
        "stop_loss_pct": 0.03,
        "take_profit_pct": 0.06,
        "max_daily_loss": 0.1
    }
}
