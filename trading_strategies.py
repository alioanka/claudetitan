"""
Advanced Trading Strategies for the Bot
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import ta
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class Signal:
    """Trading signal data structure"""
    symbol: str
    side: str  # 'long', 'short', or 'close'
    strength: float  # 0-1 signal strength
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float  # 0-1 confidence level
    strategy: str
    timestamp: datetime
    metadata: Dict[str, Any] = None

class BaseStrategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, name: str, timeframe: str = "15m"):
        self.name = name
        self.timeframe = timeframe
        self.enabled = True
        self.weight = 1.0  # Strategy weight for ensemble
    
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Optional[Signal]:
        """Generate trading signal based on market data"""
        pass
    
    @abstractmethod
    def get_required_indicators(self) -> List[str]:
        """Return list of required indicators for this strategy"""
        pass

class ScalpingStrategy(BaseStrategy):
    """High-frequency scalping strategy using multiple timeframes"""
    
    def __init__(self):
        super().__init__("Scalping", "1m")
        self.fast_ema = 9
        self.slow_ema = 21
        self.rsi_period = 14
        self.bb_period = 20
        self.bb_std = 2
    
    def get_required_indicators(self) -> List[str]:
        return ["ema", "rsi", "bollinger_bands", "volume", "atr"]
    
    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Optional[Signal]:
        try:
            if len(data) < 50:
                return None
            
            # Calculate indicators
            data = self._calculate_indicators(data)
            
            # Get latest values
            current_price = data['close'].iloc[-1]
            fast_ema = data['ema_fast'].iloc[-1]
            slow_ema = data['ema_slow'].iloc[-1]
            rsi = data['rsi'].iloc[-1]
            bb_upper = data['bb_upper'].iloc[-1]
            bb_lower = data['bb_lower'].iloc[-1]
            bb_middle = data['bb_middle'].iloc[-1]
            atr = data['atr'].iloc[-1]
            volume = data['volume'].iloc[-1]
            avg_volume = data['volume'].rolling(20).mean().iloc[-1]
            
            # Scalping conditions
            signal_strength = 0.0
            side = None
            confidence = 0.0
            
            # Long conditions
            long_conditions = [
                fast_ema > slow_ema,  # Uptrend
                current_price > fast_ema,  # Price above fast EMA
                rsi < 70 and rsi > 30,  # RSI not overbought/oversold
                current_price < bb_upper,  # Not at upper Bollinger Band
                volume > avg_volume * 1.2,  # Above average volume
                current_price > bb_middle  # Above middle BB
            ]
            
            # Short conditions
            short_conditions = [
                fast_ema < slow_ema,  # Downtrend
                current_price < fast_ema,  # Price below fast EMA
                rsi > 30 and rsi < 70,  # RSI not overbought/oversold
                current_price > bb_lower,  # Not at lower Bollinger Band
                volume > avg_volume * 1.2,  # Above average volume
                current_price < bb_middle  # Below middle BB
            ]
            
            # Calculate signal strength
            long_score = sum(long_conditions) / len(long_conditions)
            short_score = sum(short_conditions) / len(short_conditions)
            
            if long_score > 0.6:
                side = "long"
                signal_strength = long_score
                confidence = min(0.9, long_score + 0.1)
            elif short_score > 0.6:
                side = "short"
                signal_strength = short_score
                confidence = min(0.9, short_score + 0.1)
            
            if side and signal_strength > 0.5:
                # Calculate stop loss and take profit (more reasonable levels)
                stop_loss = current_price * (0.95 if side == "long" else 1.05)  # 5% stop loss
                take_profit = current_price * (1.10 if side == "long" else 0.90)  # 10% take profit
                
                return Signal(
                    symbol=symbol,
                    side=side,
                    strength=signal_strength,
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    confidence=confidence,
                    strategy=self.name,
                    timestamp=datetime.now(),
                    metadata={
                        "rsi": rsi,
                        "bb_position": (current_price - bb_lower) / (bb_upper - bb_lower),
                        "volume_ratio": volume / avg_volume,
                        "ema_cross": fast_ema - slow_ema
                    }
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating scalping signal for {symbol}: {e}")
            return None
    
    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        # EMAs
        data['ema_fast'] = ta.trend.EMAIndicator(data['close'], window=self.fast_ema).ema_indicator()
        data['ema_slow'] = ta.trend.EMAIndicator(data['close'], window=self.slow_ema).ema_indicator()
        
        # RSI
        data['rsi'] = ta.momentum.RSIIndicator(data['close'], window=self.rsi_period).rsi()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(data['close'], window=self.bb_period, window_dev=self.bb_std)
        data['bb_upper'] = bb.bollinger_hband()
        data['bb_middle'] = bb.bollinger_mavg()
        data['bb_lower'] = bb.bollinger_lband()
        
        # ATR
        data['atr'] = ta.volatility.AverageTrueRange(data['high'], data['low'], data['close'], window=14).average_true_range()
        
        return data

class MeanReversionStrategy(BaseStrategy):
    """Mean reversion strategy using Bollinger Bands and RSI"""
    
    def __init__(self):
        super().__init__("MeanReversion", "15m")
        self.bb_period = 20
        self.bb_std = 2
        self.rsi_period = 14
        self.oversold_threshold = 30
        self.overbought_threshold = 70
    
    def get_required_indicators(self) -> List[str]:
        return ["bollinger_bands", "rsi", "volume"]
    
    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Optional[Signal]:
        try:
            if len(data) < 30:
                return None
            
            data = self._calculate_indicators(data)
            
            current_price = data['close'].iloc[-1]
            bb_upper = data['bb_upper'].iloc[-1]
            bb_lower = data['bb_lower'].iloc[-1]
            bb_middle = data['bb_middle'].iloc[-1]
            rsi = data['rsi'].iloc[-1]
            volume = data['volume'].iloc[-1]
            avg_volume = data['volume'].rolling(20).mean().iloc[-1]
            
            signal_strength = 0.0
            side = None
            confidence = 0.0
            
            # Long conditions (oversold)
            long_conditions = [
                current_price <= bb_lower,  # Price at or below lower BB
                rsi < self.oversold_threshold,  # RSI oversold
                volume > avg_volume * 0.8,  # Decent volume
                current_price < bb_middle  # Price below middle BB
            ]
            
            # Short conditions (overbought)
            short_conditions = [
                current_price >= bb_upper,  # Price at or above upper BB
                rsi > self.overbought_threshold,  # RSI overbought
                volume > avg_volume * 0.8,  # Decent volume
                current_price > bb_middle  # Price above middle BB
            ]
            
            long_score = sum(long_conditions) / len(long_conditions)
            short_score = sum(short_conditions) / len(short_conditions)
            
            if long_score > 0.7:
                side = "long"
                signal_strength = long_score
                confidence = min(0.95, long_score + 0.2)
            elif short_score > 0.7:
                side = "short"
                signal_strength = short_score
                confidence = min(0.95, short_score + 0.2)
            
            if side and signal_strength > 0.6:
                # Calculate stop loss and take profit
                bb_width = bb_upper - bb_lower
                if side == "long":
                    stop_loss = current_price * 0.95  # 5% stop loss
                    take_profit = bb_middle + (bb_width * 0.5)  # More conservative take profit
                else:
                    stop_loss = current_price * 1.05  # 5% stop loss
                    take_profit = bb_middle - (bb_width * 0.5)  # More conservative take profit
                
                return Signal(
                    symbol=symbol,
                    side=side,
                    strength=signal_strength,
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    confidence=confidence,
                    strategy=self.name,
                    timestamp=datetime.now(),
                    metadata={
                        "rsi": rsi,
                        "bb_position": (current_price - bb_lower) / (bb_upper - bb_lower),
                        "volume_ratio": volume / avg_volume,
                        "bb_width": bb_width / bb_middle
                    }
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating mean reversion signal for {symbol}: {e}")
            return None
    
    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(data['close'], window=self.bb_period, window_dev=self.bb_std)
        data['bb_upper'] = bb.bollinger_hband()
        data['bb_middle'] = bb.bollinger_mavg()
        data['bb_lower'] = bb.bollinger_lband()
        
        # RSI
        data['rsi'] = ta.momentum.RSIIndicator(data['close'], window=self.rsi_period).rsi()
        
        return data

class TrendFollowingStrategy(BaseStrategy):
    """Trend following strategy using MACD and moving averages"""
    
    def __init__(self):
        super().__init__("TrendFollowing", "1h")
        self.fast_period = 12
        self.slow_period = 26
        self.signal_period = 9
        self.ma_period = 50
    
    def get_required_indicators(self) -> List[str]:
        return ["macd", "sma", "volume"]
    
    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Optional[Signal]:
        try:
            if len(data) < 60:
                return None
            
            data = self._calculate_indicators(data)
            
            current_price = data['close'].iloc[-1]
            macd = data['macd'].iloc[-1]
            macd_signal = data['macd_signal'].iloc[-1]
            macd_histogram = data['macd_histogram'].iloc[-1]
            sma = data['sma'].iloc[-1]
            volume = data['volume'].iloc[-1]
            avg_volume = data['volume'].rolling(20).mean().iloc[-1]
            
            # Check for MACD crossover
            prev_macd = data['macd'].iloc[-2]
            prev_signal = data['macd_signal'].iloc[-2]
            
            signal_strength = 0.0
            side = None
            confidence = 0.0
            
            # Long conditions
            long_conditions = [
                macd > macd_signal,  # MACD above signal
                prev_macd <= prev_signal,  # Recent crossover
                current_price > sma,  # Price above SMA
                macd_histogram > 0,  # Positive histogram
                volume > avg_volume * 1.1  # Above average volume
            ]
            
            # Short conditions
            short_conditions = [
                macd < macd_signal,  # MACD below signal
                prev_macd >= prev_signal,  # Recent crossover
                current_price < sma,  # Price below SMA
                macd_histogram < 0,  # Negative histogram
                volume > avg_volume * 1.1  # Above average volume
            ]
            
            long_score = sum(long_conditions) / len(long_conditions)
            short_score = sum(short_conditions) / len(short_conditions)
            
            if long_score > 0.6:
                side = "long"
                signal_strength = long_score
                confidence = min(0.9, long_score + 0.1)
            elif short_score > 0.6:
                side = "short"
                signal_strength = short_score
                confidence = min(0.9, short_score + 0.1)
            
            if side and signal_strength > 0.5:
                # Calculate stop loss and take profit
                atr = data['atr'].iloc[-1] if 'atr' in data.columns else current_price * 0.02
                if side == "long":
                    stop_loss = current_price - (atr * 2)
                    take_profit = current_price + (atr * 3)
                else:
                    stop_loss = current_price + (atr * 2)
                    take_profit = current_price - (atr * 3)
                
                return Signal(
                    symbol=symbol,
                    side=side,
                    strength=signal_strength,
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    confidence=confidence,
                    strategy=self.name,
                    timestamp=datetime.now(),
                    metadata={
                        "macd": macd,
                        "macd_signal": macd_signal,
                        "macd_histogram": macd_histogram,
                        "sma_distance": (current_price - sma) / sma,
                        "volume_ratio": volume / avg_volume
                    }
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating trend following signal for {symbol}: {e}")
            return None
    
    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        # MACD
        macd = ta.trend.MACD(data['close'], window_fast=self.fast_period, window_slow=self.slow_period, window_sign=self.signal_period)
        data['macd'] = macd.macd()
        data['macd_signal'] = macd.macd_signal()
        data['macd_histogram'] = macd.macd_diff()
        
        # SMA
        data['sma'] = ta.trend.SMAIndicator(data['close'], window=self.ma_period).sma_indicator()
        
        # ATR
        data['atr'] = ta.volatility.AverageTrueRange(data['high'], data['low'], data['close'], window=14).average_true_range()
        
        return data

class StrategyEnsemble:
    """Ensemble of multiple strategies for better signal generation"""
    
    def __init__(self):
        self.strategies = [
            ScalpingStrategy(),
            MeanReversionStrategy(),
            TrendFollowingStrategy()
        ]
        self.strategy_weights = {
            "Scalping": 0.4,
            "MeanReversion": 0.3,
            "TrendFollowing": 0.3
        }
        self.min_confidence = 0.6
        self.min_agreement = 0.5  # Minimum percentage of strategies that must agree
    
    def generate_ensemble_signal(self, data: pd.DataFrame, symbol: str) -> Optional[Signal]:
        """Generate signal using ensemble of strategies"""
        try:
            signals = []
            
            # Get signals from all strategies
            for strategy in self.strategies:
                if strategy.enabled:
                    signal = strategy.generate_signal(data, symbol)
                    if signal:
                        signals.append(signal)
            
            if not signals:
                return None
            
            # Calculate ensemble signal
            if len(signals) == 1:
                return signals[0]
            
            # Weight signals by strategy weight and confidence
            weighted_signals = []
            for signal in signals:
                weight = self.strategy_weights.get(signal.strategy, 1.0)
                weighted_confidence = signal.confidence * weight
                weighted_signals.append((signal, weighted_confidence))
            
            # Sort by weighted confidence
            weighted_signals.sort(key=lambda x: x[1], reverse=True)
            
            # Check for agreement
            long_signals = [s for s, _ in weighted_signals if s.side == "long"]
            short_signals = [s for s, _ in weighted_signals if s.side == "short"]
            
            if len(long_signals) > len(short_signals):
                dominant_side = "long"
                dominant_signals = long_signals
            elif len(short_signals) > len(long_signals):
                dominant_side = "short"
                dominant_signals = short_signals
            else:
                # Equal number, use highest confidence
                best_signal = weighted_signals[0][0]
                return best_signal if best_signal.confidence >= self.min_confidence else None
            
            # Check agreement threshold
            agreement_ratio = len(dominant_signals) / len(signals)
            if agreement_ratio < self.min_agreement:
                return None
            
            # Calculate ensemble metrics
            avg_confidence = np.mean([s.confidence for s in dominant_signals])
            avg_strength = np.mean([s.strength for s in dominant_signals])
            avg_entry_price = np.mean([s.entry_price for s in dominant_signals])
            avg_stop_loss = np.mean([s.stop_loss for s in dominant_signals])
            avg_take_profit = np.mean([s.take_profit for s in dominant_signals])
            
            # Only return signal if confidence is high enough
            if avg_confidence >= self.min_confidence:
                return Signal(
                    symbol=symbol,
                    side=dominant_side,
                    strength=avg_strength,
                    entry_price=avg_entry_price,
                    stop_loss=avg_stop_loss,
                    take_profit=avg_take_profit,
                    confidence=avg_confidence,
                    strategy="Ensemble",
                    timestamp=datetime.now(),
                    metadata={
                        "agreement_ratio": agreement_ratio,
                        "strategy_count": len(dominant_signals),
                        "individual_signals": [s.strategy for s in dominant_signals]
                    }
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating ensemble signal for {symbol}: {e}")
            return None
    
    def get_required_indicators(self) -> List[str]:
        """Get all required indicators from all strategies"""
        indicators = set()
        for strategy in self.strategies:
            indicators.update(strategy.get_required_indicators())
        return list(indicators)
