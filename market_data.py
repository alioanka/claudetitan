"""
Market Data Collection and Management
"""
import logging
import asyncio
import ccxt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import time
import json
from dataclasses import dataclass, asdict

from config import settings

logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Market data structure"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: str

class MarketDataCollector:
    """Collects and manages market data from exchanges"""
    
    def __init__(self):
        self.exchange = None
        self.data_cache = {}
        self.last_update = {}
        self.initialize_exchange()
    
    def initialize_exchange(self):
        """Initialize exchange connection"""
        try:
            exchange_class = getattr(ccxt, settings.exchange_name)
            self.exchange = exchange_class({
                'apiKey': settings.api_key,
                'secret': settings.api_secret,
                'sandbox': settings.sandbox,
                'enableRateLimit': True,
                'timeout': 30000,
            })
            
            if settings.trading_mode == "paper":
                self.exchange.set_sandbox_mode(True)
            
            logger.info(f"Initialized {settings.exchange_name} exchange")
            
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")
            raise
    
    async def get_ohlcv_data(
        self, 
        symbol: str, 
        timeframe: str = "15m", 
        limit: int = 1000
    ) -> pd.DataFrame:
        """Get OHLCV data for a symbol"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{timeframe}"
            if cache_key in self.data_cache:
                cache_time = self.last_update.get(cache_key, datetime.min)
                if datetime.now() - cache_time < timedelta(minutes=1):
                    return self.data_cache[cache_key]
            
            # Fetch from exchange
            ohlcv = await self._fetch_ohlcv(symbol, timeframe, limit)
            
            if not ohlcv:
                logger.warning(f"No data received for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Cache the data
            self.data_cache[cache_key] = df
            self.last_update[cache_key] = datetime.now()
            
            logger.debug(f"Retrieved {len(df)} candles for {symbol} {timeframe}")
            return df
            
        except Exception as e:
            logger.error(f"Error getting OHLCV data for {symbol}: {e}")
            return pd.DataFrame()
    
    async def _fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> List:
        """Fetch OHLCV data from exchange"""
        try:
            # Use asyncio to run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            ohlcv = await loop.run_in_executor(
                None, 
                self.exchange.fetch_ohlcv, 
                symbol, 
                timeframe, 
                None, 
                limit
            )
            return ohlcv
        except Exception as e:
            logger.error(f"Error fetching OHLCV from exchange: {e}")
            return []
    
    async def get_ticker(self, symbol: str) -> Optional[Dict]:
        """Get current ticker data for a symbol"""
        try:
            loop = asyncio.get_event_loop()
            ticker = await loop.run_in_executor(None, self.exchange.fetch_ticker, symbol)
            return ticker
        except Exception as e:
            logger.error(f"Error getting ticker for {symbol}: {e}")
            return None
    
    async def get_orderbook(self, symbol: str, limit: int = 20) -> Optional[Dict]:
        """Get order book for a symbol"""
        try:
            loop = asyncio.get_event_loop()
            orderbook = await loop.run_in_executor(
                None, 
                self.exchange.fetch_order_book, 
                symbol, 
                limit
            )
            return orderbook
        except Exception as e:
            logger.error(f"Error getting orderbook for {symbol}: {e}")
            return None
    
    async def get_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get recent trades for a symbol"""
        try:
            loop = asyncio.get_event_loop()
            trades = await loop.run_in_executor(
                None, 
                self.exchange.fetch_trades, 
                symbol, 
                None, 
                limit
            )
            return trades
        except Exception as e:
            logger.error(f"Error getting trades for {symbol}: {e}")
            return []
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for market data"""
        try:
            if df.empty:
                return df
            
            # Ensure we have required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                logger.error("Missing required columns for technical indicators")
                return df
            
            # Moving Averages
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            
            # ATR
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            df['atr'] = true_range.rolling(window=14).mean()
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Price change indicators
            df['price_change'] = df['close'].pct_change()
            df['price_change_5'] = df['close'].pct_change(5)
            df['price_change_20'] = df['close'].pct_change(20)
            
            # Volatility
            df['volatility'] = df['price_change'].rolling(window=20).std()
            
            # Support and Resistance levels
            df['resistance'] = df['high'].rolling(window=20).max()
            df['support'] = df['low'].rolling(window=20).min()
            
            logger.debug(f"Calculated technical indicators for {len(df)} candles")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return df
    
    async def get_market_summary(self, symbols: List[str]) -> Dict:
        """Get market summary for multiple symbols"""
        try:
            summary = {}
            
            for symbol in symbols:
                try:
                    # Get ticker data
                    ticker = await self.get_ticker(symbol)
                    if not ticker:
                        continue
                    
                    # Get recent OHLCV data
                    ohlcv = await self.get_ohlcv_data(symbol, "1h", 24)
                    
                    summary[symbol] = {
                        'price': ticker.get('last', 0),
                        'change_24h': ticker.get('change', 0),
                        'change_24h_pct': ticker.get('percentage', 0),
                        'volume_24h': ticker.get('baseVolume', 0),
                        'high_24h': ticker.get('high', 0),
                        'low_24h': ticker.get('low', 0),
                        'bid': ticker.get('bid', 0),
                        'ask': ticker.get('ask', 0),
                        'spread': ticker.get('ask', 0) - ticker.get('bid', 0),
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # Add technical indicators if we have OHLCV data
                    if not ohlcv.empty:
                        ohlcv = self.calculate_technical_indicators(ohlcv)
                        latest = ohlcv.iloc[-1]
                        
                        summary[symbol].update({
                            'rsi': latest.get('rsi', 0),
                            'macd': latest.get('macd', 0),
                            'bb_position': (latest['close'] - latest['bb_lower']) / (latest['bb_upper'] - latest['bb_lower']) if 'bb_upper' in latest else 0,
                            'volatility': latest.get('volatility', 0),
                            'trend': 'up' if latest['close'] > latest['sma_20'] else 'down' if 'sma_20' in latest else 'neutral'
                        })
                    
                except Exception as e:
                    logger.error(f"Error getting market summary for {symbol}: {e}")
                    continue
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting market summary: {e}")
            return {}
    
    async def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol is tradeable"""
        try:
            markets = await self._fetch_markets()
            return symbol in markets
        except Exception as e:
            logger.error(f"Error validating symbol {symbol}: {e}")
            return False
    
    async def _fetch_markets(self) -> Dict:
        """Fetch available markets from exchange"""
        try:
            loop = asyncio.get_event_loop()
            markets = await loop.run_in_executor(None, self.exchange.load_markets)
            return markets
        except Exception as e:
            logger.error(f"Error fetching markets: {e}")
            return {}
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            'cached_symbols': len(self.data_cache),
            'last_updates': {k: v.isoformat() for k, v in self.last_update.items()},
            'cache_size_mb': sum(df.memory_usage(deep=True).sum() for df in self.data_cache.values()) / 1024 / 1024
        }
    
    def clear_cache(self):
        """Clear data cache"""
        self.data_cache.clear()
        self.last_update.clear()
        logger.info("Market data cache cleared")

class DataStorage:
    """Handles data storage and retrieval"""
    
    def __init__(self, db_url: str = None):
        self.db_url = db_url or settings.database_url
        self.connection = None
    
    async def store_market_data(self, data: List[MarketData]):
        """Store market data to database"""
        try:
            # This would be implemented with actual database storage
            # For now, we'll just log the data
            logger.info(f"Storing {len(data)} market data points")
            
            # In a real implementation, you would:
            # 1. Connect to database
            # 2. Insert/update data
            # 3. Handle duplicates and updates
            
        except Exception as e:
            logger.error(f"Error storing market data: {e}")
    
    async def get_historical_data(
        self, 
        symbol: str, 
        timeframe: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> pd.DataFrame:
        """Get historical data from storage"""
        try:
            # This would query the database for historical data
            # For now, return empty DataFrame
            logger.info(f"Retrieving historical data for {symbol} {timeframe}")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return pd.DataFrame()
