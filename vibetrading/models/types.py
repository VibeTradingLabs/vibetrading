"""
Type definitions for exchange metadata and agent configuration.
"""

from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel


class SpotMeta(BaseModel):
    """Spot market metadata."""
    asset: str
    symbol: str
    sz_decimals: int
    price_precision: int


class PerpMeta(BaseModel):
    """Perpetual market metadata."""
    asset: str
    symbol: str
    sz_decimals: int
    price_precision: int
    max_leverage: int


class AgentStatus(str, Enum):
    """Agent lifecycle states."""
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPED = "stopped"
    DELETED = "deleted"


class StrategyType(str, Enum):
    """Strategy classification types."""
    DCA = "dca"
    ARBITRAGE = "arbitrage"
    GRID = "grid"
    SCALPING = "scalping"
    SWING = "swing"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    CUSTOM = "custom"


class RiskLevel(str, Enum):
    """Risk level classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


class AgentMetadata(BaseModel):
    """Metadata for a live trading agent."""
    name: str = ""
    strategy_type: StrategyType = StrategyType.CUSTOM
    market_type: str = "perp"
    risk_level: RiskLevel = RiskLevel.MEDIUM
    exchange: str = "hyperliquid"
    version: str = "1"
    tags: List[str] = []
    trading_assets: List[str] = []
    timeframe: str = "1m"
    model: str = ""
    auto_pilot: bool = False


class AgentMetrics(BaseModel):
    """Metrics for a trading agent."""
    spot_equity: float = 0.0
    perp_equity: float = 0.0
    total_equity: float = 0.0
    spot_cash: float = 0.0
    perp_cash: float = 0.0
    margin_used: float = 0.0
    available_margin: float = 0.0
    total_pnl: float = 0.0
    total_pnl_percentage: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_trades: int = 0
    total_orders: int = 0
    leverage_effective: float = 1.0
