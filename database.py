"""
Database models and operations for trading bot
"""
import os
import asyncio
import uuid
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import UUID

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://trading_enhanced:enhanced_password@localhost:5433/trading_bot_enhanced")

# Create engine and session
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Models
class TradingAccount(Base):
    __tablename__ = "trading_accounts"
    
    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(String, unique=True, index=True, default=lambda: str(uuid.uuid4()))
    balance = Column(Float, default=10000.0)
    equity = Column(Float, default=10000.0)
    margin_used = Column(Float, default=0.0)
    free_margin = Column(Float, default=10000.0)
    total_pnl = Column(Float, default=0.0)
    daily_pnl = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class TradingPosition(Base):
    __tablename__ = "trading_positions"
    
    id = Column(Integer, primary_key=True, index=True)
    position_id = Column(String, unique=True, index=True, default=lambda: str(uuid.uuid4()))
    symbol = Column(String, index=True)
    side = Column(String)  # 'long' or 'short'
    size = Column(Float)
    entry_price = Column(Float)
    current_price = Column(Float)
    unrealized_pnl = Column(Float, default=0.0)
    unrealized_pnl_pct = Column(Float, default=0.0)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    strategy = Column(String, default="Unknown")
    status = Column(String, default="open")  # 'open', 'closed'
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    closed_at = Column(DateTime)
    closing_reason = Column(String)

class TradingOrder(Base):
    __tablename__ = "trading_orders"
    
    id = Column(Integer, primary_key=True, index=True)
    order_id = Column(String, unique=True, index=True, default=lambda: str(uuid.uuid4()))
    symbol = Column(String, index=True)
    side = Column(String)  # 'buy' or 'sell'
    order_type = Column(String)  # 'market', 'limit', 'stop'
    amount = Column(Float)
    price = Column(Float)
    stop_price = Column(Float)
    status = Column(String, default="pending")  # 'pending', 'filled', 'cancelled'
    filled_price = Column(Float)
    filled_amount = Column(Float)
    fees = Column(Float, default=0.0)
    strategy = Column(String, default="Unknown")
    created_at = Column(DateTime, default=datetime.utcnow)
    filled_at = Column(DateTime)
    order_metadata = Column(JSON)

class TradingTrade(Base):
    __tablename__ = "trading_trades"
    
    id = Column(Integer, primary_key=True, index=True)
    trade_id = Column(String, unique=True, index=True, default=lambda: str(uuid.uuid4()))
    symbol = Column(String, index=True)
    side = Column(String)  # 'long' or 'short'
    size = Column(Float)
    entry_price = Column(Float)
    exit_price = Column(Float)
    pnl = Column(Float)
    pnl_pct = Column(Float)
    strategy = Column(String, default="Unknown")
    duration_minutes = Column(Integer)
    opened_at = Column(DateTime)
    closed_at = Column(DateTime)
    closing_reason = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

# Database operations
class DatabaseManager:
    def __init__(self):
        self.engine = engine
        self.SessionLocal = SessionLocal
    
    def create_tables(self):
        """Create all database tables"""
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self):
        """Get database session"""
        return self.SessionLocal()
    
    # Account operations
    async def save_account(self, account_data: Dict[str, Any]) -> str:
        """Save or update account data"""
        with self.get_session() as session:
            # Get or create account
            account = session.query(TradingAccount).filter(
                TradingAccount.account_id == account_data.get('account_id', 'default')
            ).first()
            
            if not account:
                account = TradingAccount(**account_data)
                session.add(account)
            else:
                for key, value in account_data.items():
                    setattr(account, key, value)
                account.updated_at = datetime.utcnow()
            
            session.commit()
            return account.account_id
    
    async def get_account(self, account_id: str = "default") -> Optional[Dict[str, Any]]:
        """Get account data"""
        with self.get_session() as session:
            account = session.query(TradingAccount).filter(
                TradingAccount.account_id == account_id
            ).first()
            
            if account:
                return {
                    'account_id': account.account_id,
                    'balance': account.balance,
                    'equity': account.equity,
                    'margin_used': account.margin_used,
                    'free_margin': account.free_margin,
                    'total_pnl': account.total_pnl,
                    'daily_pnl': account.daily_pnl,
                    'created_at': account.created_at,
                    'updated_at': account.updated_at
                }
            return None
    
    # Position operations
    async def save_position(self, position_data: Dict[str, Any]) -> str:
        """Save or update position"""
        with self.get_session() as session:
            # Check if position exists
            position = session.query(TradingPosition).filter(
                TradingPosition.position_id == position_data.get('position_id')
            ).first()
            
            if not position:
                position = TradingPosition(**position_data)
                session.add(position)
                print(f"✅ Database: New position created - {position_data.get('symbol')} {position_data.get('side')}")
            else:
                for key, value in position_data.items():
                    setattr(position, key, value)
                position.updated_at = datetime.utcnow()
                print(f"✅ Database: Position updated - {position_data.get('symbol')} {position_data.get('side')}")
            
            session.commit()
            return position.position_id
    
    async def get_positions(self, status: str = "open") -> List[Dict[str, Any]]:
        """Get positions by status"""
        with self.get_session() as session:
            positions = session.query(TradingPosition).filter(
                TradingPosition.status == status
            ).all()
            
            return [
                {
                    'position_id': pos.position_id,
                    'symbol': pos.symbol,
                    'side': pos.side,
                    'size': pos.size,
                    'entry_price': pos.entry_price,
                    'current_price': pos.current_price,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'unrealized_pnl_pct': pos.unrealized_pnl_pct,
                    'stop_loss': pos.stop_loss,
                    'take_profit': pos.take_profit,
                    'strategy': pos.strategy,
                    'status': pos.status,
                    'created_at': pos.created_at,
                    'updated_at': pos.updated_at,
                    'closed_at': pos.closed_at,
                    'closing_reason': pos.closing_reason
                }
                for pos in positions
            ]
    
    async def close_position(self, position_id: str, exit_price: float, reason: str = "manual"):
        """Close a position"""
        with self.get_session() as session:
            position = session.query(TradingPosition).filter(
                TradingPosition.position_id == position_id
            ).first()
            
            if position:
                position.status = "closed"
                position.current_price = exit_price
                position.closed_at = datetime.utcnow()
                position.closing_reason = reason
                position.updated_at = datetime.utcnow()
                
                # Calculate final P&L
                if position.side == "long":
                    position.unrealized_pnl = (exit_price - position.entry_price) * position.size
                else:
                    position.unrealized_pnl = (position.entry_price - exit_price) * position.size
                
                position.unrealized_pnl_pct = position.unrealized_pnl / (position.entry_price * position.size)
                
                session.commit()
    
    # Order operations
    async def save_order(self, order_data: Dict[str, Any]) -> str:
        """Save order"""
        with self.get_session() as session:
            order = TradingOrder(**order_data)
            session.add(order)
            session.commit()
            print(f"✅ Database: Order saved - {order_data.get('symbol')} {order_data.get('side')} {order_data.get('amount')}")
            return order.order_id
    
    async def get_orders(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent orders"""
        with self.get_session() as session:
            orders = session.query(TradingOrder).order_by(
                TradingOrder.created_at.desc()
            ).limit(limit).all()
            
            return [
                {
                    'order_id': order.order_id,
                    'symbol': order.symbol,
                    'side': order.side,
                    'order_type': order.order_type,
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
                }
                for order in orders
            ]
    
    # Trade operations
    async def save_trade(self, trade_data: Dict[str, Any]) -> str:
        """Save completed trade"""
        with self.get_session() as session:
            trade = TradingTrade(**trade_data)
            session.add(trade)
            session.commit()
            return trade.trade_id
    
    async def get_trades(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent trades"""
        with self.get_session() as session:
            trades = session.query(TradingTrade).order_by(
                TradingTrade.closed_at.desc()
            ).limit(limit).all()
            
            return [
                {
                    'trade_id': trade.trade_id,
                    'symbol': trade.symbol,
                    'side': trade.side,
                    'size': trade.size,
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price,
                    'pnl': trade.pnl,
                    'pnl_pct': trade.pnl_pct,
                    'strategy': trade.strategy,
                    'duration_minutes': trade.duration_minutes,
                    'opened_at': trade.opened_at,
                    'closed_at': trade.closed_at,
                    'closing_reason': trade.closing_reason,
                    'created_at': trade.created_at
                }
                for trade in trades
            ]

# Global database manager instance
db_manager = DatabaseManager()

# Initialize database on import
def init_database():
    """Initialize database tables"""
    try:
        db_manager.create_tables()
        print("✅ Database tables created successfully")
    except Exception as e:
        print(f"❌ Error creating database tables: {e}")

# Initialize on import
init_database()
