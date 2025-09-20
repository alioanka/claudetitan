"""
Paper Trading System for Backtesting and Strategy Testing
"""
import logging
import asyncio
import json
import uuid
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from enum import Enum

from risk_management import RiskManager, PositionRisk, PortfolioRisk
from trading_strategies import Signal, StrategyEnsemble
from market_data import MarketDataCollector
from database import db_manager

logger = logging.getLogger(__name__)

class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"

@dataclass
class PaperOrder:
    """Paper trading order"""
    id: str
    symbol: str
    side: OrderSide
    type: OrderType
    amount: float
    price: Optional[float]
    stop_price: Optional[float]
    status: OrderStatus
    created_at: datetime
    filled_at: Optional[datetime]
    filled_price: Optional[float]
    filled_amount: Optional[float]
    fees: float
    strategy: str = "Unknown"
    metadata: Dict = None

@dataclass
class PaperPosition:
    """Paper trading position"""
    symbol: str
    side: str  # 'long' or 'short'
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    created_at: datetime
    updated_at: datetime
    strategy: str = "Unknown"
    metadata: Dict = None

@dataclass
class PaperAccount:
    """Paper trading account"""
    balance: float
    equity: float
    margin_used: float
    free_margin: float
    total_pnl: float
    daily_pnl: float
    positions: List[PaperPosition]
    orders: List[PaperOrder]
    created_at: datetime
    updated_at: datetime

class PaperTradingEngine:
    """Paper trading engine for backtesting and strategy testing"""
    
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.account_id = "default"
        
        self.account = PaperAccount(
            balance=initial_balance,
            equity=initial_balance,
            margin_used=0.0,
            free_margin=initial_balance,
            total_pnl=0.0,
            daily_pnl=0.0,
            positions=[],
            orders=[],
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        self.risk_manager = RiskManager()
        self.strategy_ensemble = StrategyEnsemble()
        self.market_data_collector = MarketDataCollector()
        
        self.order_counter = 0
        self.trade_history = []
        self.performance_metrics = {}
        
        # Data collection for ML
        self.feature_data = []
        self.signal_history = []
        self.trade_outcomes = []
        
        # Cooldown mechanism to prevent overtrading
        self.last_trade_time = {}
        self.trade_cooldown = 300  # 5 minutes cooldown between trades for same symbol
    
    async def _load_account_from_db(self):
        """Load account data from database"""
        try:
            account_data = await db_manager.get_account(self.account_id)
            if account_data:
                self.account.balance = account_data['balance']
                self.account.equity = account_data['equity']
                self.account.margin_used = account_data['margin_used']
                self.account.free_margin = account_data['free_margin']
                self.account.total_pnl = account_data['total_pnl']
                self.account.daily_pnl = account_data['daily_pnl']
                logger.info(f"Loaded account from database: Balance=${self.account.balance:.2f}")
            else:
                # Create new account in database
                await db_manager.save_account({
                    'account_id': self.account_id,
                    'balance': self.account.balance,
                    'equity': self.account.equity,
                    'margin_used': self.account.margin_used,
                    'free_margin': self.account.free_margin,
                    'total_pnl': self.account.total_pnl,
                    'daily_pnl': self.account.daily_pnl
                })
                logger.info("Created new account in database")
        except Exception as e:
            logger.error(f"Error loading account from database: {e}")
    
    async def _load_positions_from_db(self):
        """Load positions from database"""
        try:
            positions_data = await db_manager.get_positions("open")
            self.account.positions = []
            
            for pos_data in positions_data:
                position = PaperPosition(
                    symbol=pos_data['symbol'],
                    side=pos_data['side'],
                    size=pos_data['size'],
                    entry_price=pos_data['entry_price'],
                    current_price=pos_data['current_price'],
                    unrealized_pnl=pos_data['unrealized_pnl'],
                    unrealized_pnl_pct=pos_data['unrealized_pnl_pct'],
                    stop_loss=pos_data['stop_loss'],
                    take_profit=pos_data['take_profit'],
                    created_at=pos_data['created_at'],
                    updated_at=pos_data['updated_at'],
                    strategy=pos_data['strategy']
                )
                self.account.positions.append(position)
            
            logger.info(f"Loaded {len(self.account.positions)} positions from database")
        except Exception as e:
            logger.error(f"Error loading positions from database: {e}")
    
    async def _save_account_to_db(self):
        """Save account data to database"""
        try:
            await db_manager.save_account({
                'account_id': self.account_id,
                'balance': self.account.balance,
                'equity': self.account.equity,
                'margin_used': self.account.margin_used,
                'free_margin': self.account.free_margin,
                'total_pnl': self.account.total_pnl,
                'daily_pnl': self.account.daily_pnl
            })
        except Exception as e:
            logger.error(f"Error saving account to database: {e}")
    
    async def _save_position_to_db(self, position: PaperPosition):
        """Save position to database"""
        try:
            position_id = str(uuid.uuid4())
            await db_manager.save_position({
                'position_id': position_id,
                'symbol': position.symbol,
                'side': position.side,
                'size': position.size,
                'entry_price': position.entry_price,
                'current_price': position.current_price,
                'unrealized_pnl': position.unrealized_pnl,
                'unrealized_pnl_pct': position.unrealized_pnl_pct,
                'stop_loss': position.stop_loss,
                'take_profit': position.take_profit,
                'strategy': position.strategy,
                'status': 'open',
                'created_at': position.created_at,
                'updated_at': position.updated_at
            })
            return position_id
        except Exception as e:
            logger.error(f"Error saving position to database: {e}")
            return None
    
    async def _save_order_to_db(self, order: PaperOrder):
        """Save order to database"""
        try:
            order_id = str(uuid.uuid4())
            await db_manager.save_order({
                'order_id': order_id,
                'symbol': order.symbol,
                'side': order.side,
                'order_type': order.type,
                'amount': order.amount,
                'price': order.price,
                'stop_price': order.stop_price,
                'status': order.status,
                'filled_price': order.filled_price,
                'filled_amount': order.filled_amount,
                'fees': order.fees,
                'strategy': order.strategy,
                'created_at': order.created_at,
                'filled_at': order.filled_at,
                'order_metadata': order.metadata
            })
            return order_id
        except Exception as e:
            logger.error(f"Error saving order to database: {e}")
            return None
        
        logger.info(f"Initialized paper trading engine with ${initial_balance:,.2f}")
    
    async def initialize(self):
        """Initialize the trading engine with data from database"""
        await self._load_account_from_db()
        await self._load_positions_from_db()
        
        # Update account metrics after loading
        await self._update_account_metrics()
        
        logger.info(f"Trading engine initialized with database data: {len(self.account.positions)} positions, Balance=${self.account.balance:.2f}")
    
    def generate_order_id(self) -> str:
        """Generate unique order ID"""
        self.order_counter += 1
        return f"paper_{self.order_counter}_{int(datetime.now().timestamp())}"
    
    async def place_order(
        self, 
        symbol: str, 
        side: OrderSide, 
        amount: float, 
        price: Optional[float] = None,
        order_type: OrderType = OrderType.MARKET,
        stop_price: Optional[float] = None,
        strategy: str = "Unknown"
    ) -> PaperOrder:
        """Place a paper trading order"""
        try:
            # Generate order ID
            order_id = self.generate_order_id()
            
            # Create order
            order = PaperOrder(
                id=order_id,
                symbol=symbol,
                side=side,
                type=order_type,
                amount=amount,
                price=price,
                stop_price=stop_price,
                status=OrderStatus.PENDING,
                created_at=datetime.now(),
                filled_at=None,
                filled_price=None,
                filled_amount=None,
                fees=0.0,
                strategy=strategy,
                metadata={}
            )
            
            # Add to orders list
            self.account.orders.append(order)
            
            # Try to fill the order immediately for market orders
            if order_type == OrderType.MARKET:
                await self._fill_market_order(order)
            elif order_type == OrderType.LIMIT:
                await self._check_limit_order(order)
            elif order_type in [OrderType.STOP_LOSS, OrderType.TAKE_PROFIT]:
                # Stop orders are checked during price updates
                pass
            
            # Save order to database
            await self._save_order_to_db(order)
            
            logger.info(f"Placed {order_type.value} order: {side.value} {amount} {symbol} at {price}")
            return order
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None
    
    async def _fill_market_order(self, order: PaperOrder):
        """Fill a market order at current market price"""
        try:
            # Get current market price
            ticker = await self.market_data_collector.get_ticker(order.symbol)
            if not ticker:
                order.status = OrderStatus.REJECTED
                return
            
            # Determine fill price
            if order.side == OrderSide.BUY:
                fill_price = ticker.get('ask', ticker.get('last', 0))
            else:
                fill_price = ticker.get('bid', ticker.get('last', 0))
            
            if fill_price <= 0:
                order.status = OrderStatus.REJECTED
                return
            
            # Calculate fees (0.1% for paper trading)
            fees = order.amount * fill_price * 0.001
            total_cost = (order.amount * fill_price) + fees
            
            # Check if we have enough balance
            if order.side == OrderSide.BUY and total_cost > self.account.free_margin:
                order.status = OrderStatus.REJECTED
                logger.warning(f"Insufficient balance for order {order.id}")
                return
            
            # Fill the order
            order.status = OrderStatus.FILLED
            order.filled_at = datetime.now()
            order.filled_price = fill_price
            order.filled_amount = order.amount
            order.fees = fees
            
            # Update account
            await self._update_account_after_fill(order)
            
            # Create or update position
            await self._update_position_after_fill(order)
            
            logger.info(f"Filled market order {order.id}: {order.side.value} {order.amount} {order.symbol} at {fill_price}")
            
        except Exception as e:
            logger.error(f"Error filling market order: {e}")
            order.status = OrderStatus.REJECTED
    
    async def _check_limit_order(self, order: PaperOrder):
        """Check if limit order should be filled"""
        try:
            if order.status != OrderStatus.PENDING:
                return
            
            ticker = await self.market_data_collector.get_ticker(order.symbol)
            if not ticker:
                return
            
            current_price = ticker.get('last', 0)
            
            # Check if limit price is reached
            should_fill = False
            if order.side == OrderSide.BUY and current_price <= order.price:
                should_fill = True
            elif order.side == OrderSide.SELL and current_price >= order.price:
                should_fill = True
            
            if should_fill:
                await self._fill_market_order(order)
                
        except Exception as e:
            logger.error(f"Error checking limit order: {e}")
    
    async def _update_account_after_fill(self, order: PaperOrder):
        """Update account after order fill"""
        try:
            # For paper trading, we don't actually deduct/add money from balance
            # The balance only changes when positions are closed with P&L
            # This is just for tracking the order
            
            # Update equity and margins
            await self._update_account_metrics()
            
        except Exception as e:
            logger.error(f"Error updating account after fill: {e}")
    
    async def _update_position_after_fill(self, order: PaperOrder):
        """Update position after order fill"""
        try:
            # Find existing position
            existing_position = None
            for pos in self.account.positions:
                if pos.symbol == order.symbol:
                    existing_position = pos
                    break
            
            if existing_position:
                # Update existing position
                if order.side == OrderSide.BUY:
                    # Adding to long position or reducing short position
                    if existing_position.side == "long":
                        # Add to long position
                        total_size = existing_position.size + order.filled_amount
                        avg_price = ((existing_position.size * existing_position.entry_price) + 
                                   (order.filled_amount * order.filled_price)) / total_size
                        existing_position.size = total_size
                        existing_position.entry_price = avg_price
                    else:
                        # Reduce short position
                        existing_position.size -= order.filled_amount
                        if existing_position.size <= 0:
                            # Close short position, open long
                            existing_position.side = "long"
                            existing_position.size = abs(existing_position.size)
                            existing_position.entry_price = order.filled_price
                else:
                    # Adding to short position or reducing long position
                    if existing_position.side == "short":
                        # Add to short position
                        total_size = existing_position.size + order.filled_amount
                        avg_price = ((existing_position.size * existing_position.entry_price) + 
                                   (order.filled_amount * order.filled_price)) / total_size
                        existing_position.size = total_size
                        existing_position.entry_price = avg_price
                    else:
                        # Reduce long position
                        existing_position.size -= order.filled_amount
                        if existing_position.size <= 0:
                            # Close long position, open short
                            existing_position.side = "short"
                            existing_position.size = abs(existing_position.size)
                            existing_position.entry_price = order.filled_price
                
                existing_position.updated_at = datetime.now()
            else:
                # Create new position
                position = PaperPosition(
                    symbol=order.symbol,
                    side="long" if order.side == OrderSide.BUY else "short",
                    size=order.filled_amount,
                    entry_price=order.filled_price,
                    current_price=order.filled_price,
                    unrealized_pnl=0.0,
                    unrealized_pnl_pct=0.0,
                    stop_loss=None,
                    take_profit=None,
                    created_at=datetime.utcnow(),  # Use UTC time
                    updated_at=datetime.utcnow(),  # Use UTC time
                    strategy=order.strategy
                )
                self.account.positions.append(position)
                
                # Save position to database
                await self._save_position_to_db(position)
            
            # Remove positions with zero size
            self.account.positions = [pos for pos in self.account.positions if pos.size > 0]
            
        except Exception as e:
            logger.error(f"Error updating position after fill: {e}")
    
    async def _update_account_metrics(self):
        """Update account metrics"""
        try:
            # Update position prices and PnL
            total_unrealized_pnl = 0.0
            
            for position in self.account.positions:
                # Get current price
                ticker = await self.market_data_collector.get_ticker(position.symbol)
                if ticker:
                    position.current_price = ticker.get('last', position.current_price)
                
                # Calculate unrealized PnL
                if position.side == "long":
                    position.unrealized_pnl = (position.current_price - position.entry_price) * position.size
                else:
                    position.unrealized_pnl = (position.entry_price - position.current_price) * position.size
                
                position.unrealized_pnl_pct = position.unrealized_pnl / (position.entry_price * position.size)
                total_unrealized_pnl += position.unrealized_pnl
                position.updated_at = datetime.now()
            
            # Update account metrics
            self.account.equity = self.account.balance + total_unrealized_pnl
            
            # Calculate total P&L from realized trades
            realized_pnl = sum(trade['pnl'] for trade in self.trade_outcomes)
            self.account.total_pnl = realized_pnl + total_unrealized_pnl
            
            self.account.daily_pnl = total_unrealized_pnl  # Simplified for paper trading
            self.account.updated_at = datetime.now()
            
        except Exception as e:
            logger.error(f"Error updating account metrics: {e}")
    
    async def check_stop_orders(self):
        """Check and execute stop loss and take profit orders"""
        try:
            for position in self.account.positions:
                if not position.stop_loss and not position.take_profit:
                    continue
                
                # Get current price
                ticker = await self.market_data_collector.get_ticker(position.symbol)
                if not ticker:
                    continue
                
                current_price = ticker.get('last', 0)
                should_close = False
                close_reason = ""
                
                # Check stop loss
                if position.stop_loss:
                    if position.side == "long" and current_price <= position.stop_loss:
                        should_close = True
                        close_reason = "stop_loss"
                    elif position.side == "short" and current_price >= position.stop_loss:
                        should_close = True
                        close_reason = "stop_loss"
                
                # Check take profit
                if position.take_profit and not should_close:
                    if position.side == "long" and current_price >= position.take_profit:
                        should_close = True
                        close_reason = "take_profit"
                    elif position.side == "short" and current_price <= position.take_profit:
                        should_close = True
                        close_reason = "take_profit"
                
                if should_close:
                    await self.close_position(position, close_reason)
                    
        except Exception as e:
            logger.error(f"Error checking stop orders: {e}")
    
    async def close_position(self, position: PaperPosition, reason: str = "manual"):
        """Close a position"""
        try:
            # Create closing order
            side = OrderSide.SELL if position.side == "long" else OrderSide.BUY
            order = await self.place_order(
                symbol=position.symbol,
                side=side,
                amount=position.size,
                order_type=OrderType.MARKET,
                strategy=getattr(position, 'strategy', 'Manual')
            )
            
            if order and order.status == OrderStatus.FILLED:
                # Update balance with realized P&L (convert unrealized to realized)
                realized_pnl = position.unrealized_pnl
                self.account.balance += realized_pnl
                
                # Save trade to database
                await db_manager.save_trade({
                    'symbol': position.symbol,
                    'side': position.side,
                    'size': position.size,
                    'entry_price': position.entry_price,
                    'exit_price': order.filled_price,
                    'pnl': realized_pnl,
                    'pnl_pct': position.unrealized_pnl_pct,
                    'strategy': position.strategy,
                    'duration_minutes': int((datetime.now() - position.created_at).total_seconds() / 60),
                    'opened_at': position.created_at,
                    'closed_at': datetime.now(),
                    'closing_reason': reason
                })
                
                # Close position in database
                await db_manager.close_position(position.position_id if hasattr(position, 'position_id') else None, order.filled_price, reason)
                
                # Record trade outcome for ML
                self.trade_outcomes.append({
                    'symbol': position.symbol,
                    'side': position.side,
                    'entry_price': position.entry_price,
                    'exit_price': order.filled_price,
                    'size': position.size,
                    'pnl': position.unrealized_pnl,
                    'pnl_pct': position.unrealized_pnl_pct,
                    'duration': (datetime.now() - position.created_at).total_seconds(),
                    'reason': reason,
                    'timestamp': datetime.now()
                })
                
                logger.info(f"Closed position {position.symbol} {position.side}: PnL ${position.unrealized_pnl:.2f} ({position.unrealized_pnl_pct:.2%})")
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
    
    async def run_strategy_cycle(self, symbols: List[str]):
        """Run one cycle of strategy execution"""
        try:
            # Update account metrics
            await self._update_account_metrics()
            
            # Check stop orders
            await self.check_stop_orders()
            
            # Check limit orders
            for order in self.account.orders:
                if order.status == OrderStatus.PENDING and order.type == OrderType.LIMIT:
                    await self._check_limit_order(order)
            
            # Generate signals for each symbol
            for symbol in symbols:
                try:
                    # Get market data
                    data = await self.market_data_collector.get_ohlcv_data(symbol, "15m", 100)
                    if data.empty:
                        continue
                    
                    # Calculate technical indicators
                    data = self.market_data_collector.calculate_technical_indicators(data)
                    
                    # Generate signal
                    signal = self.strategy_ensemble.generate_ensemble_signal(data, symbol)
                    if not signal:
                        logger.debug(f"No signal generated for {symbol}")
                        continue
                    
                    logger.info(f"Generated signal for {symbol}: {signal.strategy} - {signal.side} at {signal.entry_price}")
                    
                    # Store signal for ML training
                    self.signal_history.append({
                        'symbol': symbol,
                        'signal': asdict(signal),
                        'timestamp': datetime.now(),
                        'market_data': data.iloc[-1].to_dict()
                    })
                    
                    # Check cooldown period
                    current_time = datetime.now().timestamp()
                    last_trade = self.last_trade_time.get(symbol, 0)
                    if current_time - last_trade < self.trade_cooldown:
                        logger.debug(f"Symbol {symbol} in cooldown period, skipping trade")
                        continue
                    
                    # Check if we should open a position
                    current_positions = [asdict(pos) for pos in self.account.positions]
                    should_open, reason = self.risk_manager.should_open_position(
                        symbol, signal.side, current_positions, self.account.equity
                    )
                    
                    if should_open:
                        # Calculate position size
                        position_size, leverage = self.risk_manager.calculate_position_size(
                            symbol, signal.entry_price, signal.stop_loss, self.account.equity
                        )
                        
                        if position_size > 0:
                            # Place order
                            side = OrderSide.BUY if signal.side == "long" else OrderSide.SELL
                            order = await self.place_order(
                                symbol=symbol,
                                side=side,
                                amount=position_size,
                                order_type=OrderType.MARKET,
                                strategy=signal.strategy
                            )
                            
                            logger.info(f"Placed order for {symbol} with strategy: {signal.strategy}")
                            
                            if order and order.status == OrderStatus.FILLED:
                                # Set stop loss and take profit
                                for pos in self.account.positions:
                                    if pos.symbol == symbol:
                                        pos.stop_loss = signal.stop_loss
                                        pos.take_profit = signal.take_profit
                                        break
                                
                                # Update cooldown time
                                self.last_trade_time[symbol] = current_time
                                
                                logger.info(f"Opened {signal.side} position in {symbol}: {position_size} at {signal.entry_price}")
                    else:
                        logger.debug(f"Signal rejected for {symbol}: {reason}")
                
                except Exception as e:
                    logger.error(f"Error processing symbol {symbol}: {e}")
                    continue
            
            # Store feature data for ML
            await self._collect_feature_data(symbols)
            
        except Exception as e:
            logger.error(f"Error in strategy cycle: {e}")
    
    async def _collect_feature_data(self, symbols: List[str]):
        """Collect feature data for ML training"""
        try:
            for symbol in symbols:
                data = await self.market_data_collector.get_ohlcv_data(symbol, "15m", 50)
                if not data.empty:
                    data = self.market_data_collector.calculate_technical_indicators(data)
                    
                    # Extract features
                    features = {
                        'symbol': symbol,
                        'timestamp': datetime.now(),
                        'price': data['close'].iloc[-1],
                        'volume': data['volume'].iloc[-1],
                        'rsi': data.get('rsi', 0).iloc[-1],
                        'macd': data.get('macd', 0).iloc[-1],
                        'bb_position': 0,  # Calculate if BB available
                        'volatility': data.get('volatility', 0).iloc[-1],
                        'trend': 1 if data['close'].iloc[-1] > data.get('sma_20', 0).iloc[-1] else -1
                    }
                    
                    self.feature_data.append(features)
            
            # Keep only last 1000 feature records
            if len(self.feature_data) > 1000:
                self.feature_data = self.feature_data[-1000:]
                
        except Exception as e:
            logger.error(f"Error collecting feature data: {e}")
    
    def get_performance_metrics(self) -> Dict:
        """Calculate performance metrics"""
        try:
            # Use only closed trades for performance metrics
            all_trades = self.trade_outcomes.copy()
            
            if not all_trades:
                return {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0,
                    'total_pnl': 0,
                    'avg_win': 0,
                    'avg_loss': 0,
                    'max_drawdown': 0,
                    'sharpe_ratio': 0,
                    'current_equity': self.account.equity,
                    'total_balance': self.account.balance,
                    'available_balance': self.account.balance,
                    'total_return': 0,
                    'active_positions': len(self.account.positions),
                    'initial_balance': self.initial_balance,
                    'daily_pnl': self.account.daily_pnl,
                    'weekly_pnl': 0,  # Will calculate
                    'monthly_pnl': 0  # Will calculate
                }
            
            # Basic metrics
            total_trades = len(all_trades)
            winning_trades = len([t for t in all_trades if t['pnl'] > 0])
            losing_trades = len([t for t in all_trades if t['pnl'] < 0])
            
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0  # Convert to percentage
            
            # PnL metrics
            total_pnl = sum(t['pnl'] for t in all_trades)
            avg_win = np.mean([t['pnl'] for t in all_trades if t['pnl'] > 0]) if winning_trades > 0 else 0
            avg_loss = np.mean([t['pnl'] for t in all_trades if t['pnl'] < 0]) if losing_trades > 0 else 0
            
            # Calculate proper max drawdown (peak-to-trough decline)
            max_drawdown = 0
            if all_trades:
                cumulative_pnl = 0
                peak = 0
                for trade in all_trades:
                    cumulative_pnl += trade['pnl']
                    peak = max(peak, cumulative_pnl)
                    drawdown = peak - cumulative_pnl
                    max_drawdown = max(max_drawdown, drawdown)
            
            # Calculate Sharpe ratio properly
            if len(all_trades) > 1:
                returns = [t['pnl'] for t in all_trades]
                mean_return = np.mean(returns)
                std_return = np.std(returns)
                sharpe_ratio = mean_return / std_return if std_return > 0 else 0
            else:
                sharpe_ratio = 0
            
            # Calculate daily, weekly, monthly P&L
            now = datetime.now()
            daily_trades = [t for t in all_trades if (now - t['timestamp']).days < 1]
            weekly_trades = [t for t in all_trades if (now - t['timestamp']).days < 7]
            monthly_trades = [t for t in all_trades if (now - t['timestamp']).days < 30]
            
            daily_pnl = sum(t['pnl'] for t in daily_trades)
            weekly_pnl = sum(t['pnl'] for t in weekly_trades)
            monthly_pnl = sum(t['pnl'] for t in monthly_trades)
            
            # Calculate realistic total return based on P&L
            total_return = (total_pnl / self.initial_balance) * 100 if self.initial_balance > 0 else 0
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'current_equity': self.account.equity,
                'total_balance': self.account.balance,
                'available_balance': self.account.balance,
                'total_return': total_return,
                'active_positions': len(self.account.positions),
                'initial_balance': self.initial_balance,
                'daily_pnl': daily_pnl,
                'weekly_pnl': weekly_pnl,
                'monthly_pnl': monthly_pnl
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    def get_account_summary(self) -> Dict:
        """Get account summary with JSON-serializable data"""
        def serialize_datetime(obj):
            """Convert datetime objects to ISO format strings"""
            if hasattr(obj, 'isoformat'):
                return obj.isoformat()
            return obj
        
        def serialize_position(pos):
            """Serialize position with datetime handling"""
            pos_dict = asdict(pos)
            for key, value in pos_dict.items():
                if isinstance(value, datetime):
                    pos_dict[key] = value.isoformat()
            return pos_dict
        
        def serialize_order(order):
            """Serialize order with datetime handling"""
            order_dict = asdict(order)
            for key, value in order_dict.items():
                if isinstance(value, datetime):
                    order_dict[key] = value.isoformat()
            return order_dict
        
        return {
            'balance': self.account.balance,
            'equity': self.account.equity,
            'total_pnl': self.account.total_pnl,
            'daily_pnl': self.account.daily_pnl,
            'positions': [serialize_position(pos) for pos in self.account.positions],
            'orders': [serialize_order(order) for order in self.account.orders[-10:]],  # Last 10 orders
            'performance': self.get_performance_metrics()
        }
