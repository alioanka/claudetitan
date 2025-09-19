"""
Advanced Risk Management System for Trading Bot
"""
import math
import logging
from simple_logging import get_logger
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from config import settings, RISK_PARAMETERS, RiskLevel

logger = get_logger(__name__)

@dataclass
class PositionRisk:
    """Risk metrics for a single position"""
    symbol: str
    side: str  # 'long' or 'short'
    size: float
    entry_price: float
    current_price: float
    leverage: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    position_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    max_loss: float
    max_loss_pct: float

@dataclass
class PortfolioRisk:
    """Overall portfolio risk metrics"""
    total_value: float
    total_exposure: float
    total_unrealized_pnl: float
    total_unrealized_pnl_pct: float
    daily_pnl: float
    daily_pnl_pct: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    var_95: float  # Value at Risk 95%
    cvar_95: float  # Conditional Value at Risk 95%
    position_count: int
    risk_score: float  # 0-100 risk score

class RiskManager:
    """Advanced risk management system"""
    
    def __init__(self):
        self.risk_params = RISK_PARAMETERS[settings.risk_level]
        self.daily_pnl_history = []
        self.max_drawdown = 0.0
        self.peak_value = 0.0
        
    def calculate_position_size(
        self, 
        symbol: str, 
        entry_price: float, 
        stop_loss: float, 
        account_balance: float,
        volatility: float = 0.02
    ) -> Tuple[float, float]:
        """
        Calculate optimal position size using Kelly Criterion and risk management
        
        Args:
            symbol: Trading pair symbol
            entry_price: Entry price for the position
            stop_loss: Stop loss price
            account_balance: Current account balance
            volatility: Historical volatility of the asset
            
        Returns:
            Tuple of (position_size, leverage)
        """
        try:
            # Calculate risk per trade (1-2% of account)
            risk_per_trade = account_balance * 0.02
            
            # Calculate stop loss distance
            stop_loss_distance = abs(entry_price - stop_loss) / entry_price
            
            # Calculate position size based on risk
            position_size = risk_per_trade / stop_loss_distance
            
            # Apply maximum position size limit
            max_position_value = account_balance * self.risk_params["max_position_size"]
            position_size = min(position_size, max_position_value / entry_price)
            
            # Calculate optimal leverage based on volatility
            # Higher volatility = lower leverage
            base_leverage = self.risk_params["max_leverage"]
            volatility_adjusted_leverage = base_leverage * (0.02 / max(volatility, 0.01))
            leverage = min(volatility_adjusted_leverage, self.risk_params["max_leverage"])
            
            # Ensure minimum position size
            min_position_size = 0.001
            position_size = max(position_size, min_position_size)
            
            logger.info(f"Calculated position size for {symbol}: {position_size:.6f}, leverage: {leverage:.2f}")
            
            return position_size, leverage
            
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            return 0.0, 1.0
    
    def calculate_stop_loss_take_profit(
        self, 
        symbol: str, 
        entry_price: float, 
        side: str,
        volatility: float = 0.02
    ) -> Tuple[float, float]:
        """
        Calculate dynamic stop loss and take profit levels
        
        Args:
            symbol: Trading pair symbol
            entry_price: Entry price
            side: 'long' or 'short'
            volatility: Asset volatility
            
        Returns:
            Tuple of (stop_loss, take_profit)
        """
        try:
            # Base stop loss and take profit percentages
            base_stop_loss = self.risk_params["stop_loss_pct"]
            base_take_profit = self.risk_params["take_profit_pct"]
            
            # Adjust based on volatility
            volatility_multiplier = max(0.5, min(2.0, volatility / 0.02))
            
            stop_loss_pct = base_stop_loss * volatility_multiplier
            take_profit_pct = base_take_profit * volatility_multiplier
            
            if side == "long":
                stop_loss = entry_price * (1 - stop_loss_pct)
                take_profit = entry_price * (1 + take_profit_pct)
            else:  # short
                stop_loss = entry_price * (1 + stop_loss_pct)
                take_profit = entry_price * (1 - take_profit_pct)
            
            logger.info(f"Calculated SL/TP for {symbol} {side}: SL={stop_loss:.6f}, TP={take_profit:.6f}")
            
            return stop_loss, take_profit
            
        except Exception as e:
            logger.error(f"Error calculating SL/TP for {symbol}: {e}")
            return entry_price * 0.98, entry_price * 1.04
    
    def assess_position_risk(self, position: Dict) -> PositionRisk:
        """Assess risk for a single position"""
        try:
            symbol = position["symbol"]
            side = position["side"]
            size = position["size"]
            entry_price = position["entry_price"]
            current_price = position["current_price"]
            leverage = position.get("leverage", 1.0)
            
            # Calculate position metrics
            position_value = size * current_price
            unrealized_pnl = (current_price - entry_price) * size if side == "long" else (entry_price - current_price) * size
            unrealized_pnl_pct = unrealized_pnl / (size * entry_price)
            
            # Calculate stop loss and take profit
            stop_loss, take_profit = self.calculate_stop_loss_take_profit(
                symbol, entry_price, side
            )
            
            # Calculate risk metrics
            max_loss = abs(entry_price - stop_loss) * size
            max_loss_pct = max_loss / (size * entry_price)
            risk_reward_ratio = abs(take_profit - entry_price) / abs(entry_price - stop_loss)
            
            return PositionRisk(
                symbol=symbol,
                side=side,
                size=size,
                entry_price=entry_price,
                current_price=current_price,
                leverage=leverage,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=risk_reward_ratio,
                position_value=position_value,
                unrealized_pnl=unrealized_pnl,
                unrealized_pnl_pct=unrealized_pnl_pct,
                max_loss=max_loss,
                max_loss_pct=max_loss_pct
            )
            
        except Exception as e:
            logger.error(f"Error assessing position risk: {e}")
            return None
    
    def assess_portfolio_risk(self, positions: List[Dict], account_balance: float) -> PortfolioRisk:
        """Assess overall portfolio risk"""
        try:
            if not positions:
                return PortfolioRisk(
                    total_value=account_balance,
                    total_exposure=0.0,
                    total_unrealized_pnl=0.0,
                    total_unrealized_pnl_pct=0.0,
                    daily_pnl=0.0,
                    daily_pnl_pct=0.0,
                    max_drawdown=0.0,
                    max_drawdown_pct=0.0,
                    sharpe_ratio=0.0,
                    var_95=0.0,
                    cvar_95=0.0,
                    position_count=0,
                    risk_score=0.0
                )
            
            # Calculate basic metrics
            total_exposure = sum(pos["size"] * pos["current_price"] for pos in positions)
            total_unrealized_pnl = sum(
                (pos["current_price"] - pos["entry_price"]) * pos["size"] 
                if pos["side"] == "long" 
                else (pos["entry_price"] - pos["current_price"]) * pos["size"]
                for pos in positions
            )
            total_value = account_balance + total_unrealized_pnl
            total_unrealized_pnl_pct = total_unrealized_pnl / account_balance
            
            # Calculate daily PnL
            daily_pnl = total_unrealized_pnl
            daily_pnl_pct = daily_pnl / account_balance
            
            # Update drawdown tracking
            if total_value > self.peak_value:
                self.peak_value = total_value
            current_drawdown = self.peak_value - total_value
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown
            
            max_drawdown_pct = self.max_drawdown / self.peak_value if self.peak_value > 0 else 0
            
            # Calculate Sharpe ratio (simplified)
            if len(self.daily_pnl_history) > 1:
                returns = np.array(self.daily_pnl_history)
                sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            else:
                sharpe_ratio = 0.0
            
            # Calculate VaR and CVaR
            if len(self.daily_pnl_history) > 10:
                returns = np.array(self.daily_pnl_history)
                var_95 = np.percentile(returns, 5)  # 5th percentile
                cvar_95 = np.mean(returns[returns <= var_95])  # Mean of returns below VaR
            else:
                var_95 = -0.05  # Default 5% VaR
                cvar_95 = -0.08  # Default 8% CVaR
            
            # Calculate risk score (0-100, higher = riskier)
            risk_score = self._calculate_risk_score(
                total_exposure, account_balance, max_drawdown_pct, 
                len(positions), total_unrealized_pnl_pct
            )
            
            # Update daily PnL history
            self.daily_pnl_history.append(daily_pnl_pct)
            if len(self.daily_pnl_history) > 30:  # Keep last 30 days
                self.daily_pnl_history.pop(0)
            
            return PortfolioRisk(
                total_value=total_value,
                total_exposure=total_exposure,
                total_unrealized_pnl=total_unrealized_pnl,
                total_unrealized_pnl_pct=total_unrealized_pnl_pct,
                daily_pnl=daily_pnl,
                daily_pnl_pct=daily_pnl_pct,
                max_drawdown=self.max_drawdown,
                max_drawdown_pct=max_drawdown_pct,
                sharpe_ratio=sharpe_ratio,
                var_95=var_95,
                cvar_95=cvar_95,
                position_count=len(positions),
                risk_score=risk_score
            )
            
        except Exception as e:
            logger.error(f"Error assessing portfolio risk: {e}")
            return None
    
    def _calculate_risk_score(
        self, 
        total_exposure: float, 
        account_balance: float, 
        max_drawdown_pct: float,
        position_count: int, 
        daily_pnl_pct: float
    ) -> float:
        """Calculate overall risk score (0-100)"""
        try:
            # Exposure risk (0-30 points)
            exposure_ratio = total_exposure / account_balance if account_balance > 0 else 0
            exposure_risk = min(30, exposure_ratio * 30)
            
            # Drawdown risk (0-25 points)
            drawdown_risk = min(25, max_drawdown_pct * 500)
            
            # Concentration risk (0-20 points)
            concentration_risk = min(20, position_count * 2)
            
            # Daily PnL risk (0-25 points)
            daily_pnl_risk = min(25, abs(daily_pnl_pct) * 500)
            
            total_risk = exposure_risk + drawdown_risk + concentration_risk + daily_pnl_risk
            
            return min(100, max(0, total_risk))
            
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return 50.0  # Default moderate risk
    
    def should_open_position(
        self, 
        symbol: str, 
        side: str, 
        current_positions: List[Dict],
        account_balance: float
    ) -> Tuple[bool, str]:
        """
        Determine if a new position should be opened
        
        Returns:
            Tuple of (should_open, reason)
        """
        try:
            # Check maximum positions limit
            if len(current_positions) >= settings.max_positions:
                return False, "Maximum positions limit reached"
            
            # Check daily loss limit
            portfolio_risk = self.assess_portfolio_risk(current_positions, account_balance)
            if portfolio_risk.daily_pnl_pct <= -self.risk_params["max_daily_loss"]:
                return False, "Daily loss limit reached"
            
            # Check risk score
            if portfolio_risk.risk_score > 80:
                return False, "Portfolio risk too high"
            
            # Check if already have position in this symbol
            for pos in current_positions:
                if pos["symbol"] == symbol:
                    return False, "Position already exists for this symbol"
            
            return True, "Position approved"
            
        except Exception as e:
            logger.error(f"Error checking if should open position: {e}")
            return False, f"Error: {e}"
    
    def should_close_position(self, position: Dict, portfolio_risk: PortfolioRisk) -> Tuple[bool, str]:
        """
        Determine if a position should be closed
        
        Returns:
            Tuple of (should_close, reason)
        """
        try:
            # Check stop loss
            current_price = position["current_price"]
            entry_price = position["entry_price"]
            side = position["side"]
            
            if side == "long" and current_price <= position.get("stop_loss", 0):
                return True, "Stop loss triggered"
            elif side == "short" and current_price >= position.get("stop_loss", float('inf')):
                return True, "Stop loss triggered"
            
            # Check take profit
            if side == "long" and current_price >= position.get("take_profit", 0):
                return True, "Take profit triggered"
            elif side == "short" and current_price <= position.get("take_profit", float('inf')):
                return True, "Take profit triggered"
            
            # Check portfolio risk
            if portfolio_risk.risk_score > 90:
                return True, "Portfolio risk too high"
            
            # Check daily loss limit
            if portfolio_risk.daily_pnl_pct <= -self.risk_params["max_daily_loss"]:
                return True, "Daily loss limit reached"
            
            return False, "Position should remain open"
            
        except Exception as e:
            logger.error(f"Error checking if should close position: {e}")
            return False, f"Error: {e}"
