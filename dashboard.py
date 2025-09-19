"""
Web Dashboard for Trading Bot
"""
import logging
from simple_logging import get_logger
import asyncio
import json
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, status, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from pathlib import Path
import uvicorn
from contextlib import asynccontextmanager

from config import settings
from paper_trading import PaperTradingEngine
from simple_logging import setup_simple_logging

# Setup logging
setup_simple_logging(settings.log_level, "logs")
from market_data import MarketDataCollector
from ml_module import MLModelTrainer
from risk_management import RiskManager

logger = get_logger(__name__)

# Pydantic models for API
class LoginRequest(BaseModel):
    username: str
    password: str

class TradingConfig(BaseModel):
    trading_mode: str
    risk_level: str
    max_positions: int
    max_daily_loss: float
    max_position_size: float

class SignalResponse(BaseModel):
    symbol: str
    side: str
    strength: float
    confidence: float
    strategy: str
    timestamp: datetime

# Global variables
trading_engine = None
market_data_collector = None
ml_trainer = None
risk_manager = None
connected_clients = []

class ConnectionManager:
    """Manage WebSocket connections"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
    
    async def broadcast(self, message: str):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting message: {e}")
                disconnected.append(connection)
        
        # Remove disconnected connections
        for connection in disconnected:
            self.disconnect(connection)

manager = ConnectionManager()

# Security
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token"""
    # In production, implement proper JWT verification
    if credentials.credentials != "your-secret-token":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    return credentials.credentials

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global trading_engine, market_data_collector, ml_trainer, risk_manager
    
    # Startup
    logger.info("Starting trading bot dashboard...")
    
    # Initialize components
    trading_engine = PaperTradingEngine(initial_balance=10000.0)
    market_data_collector = MarketDataCollector()
    ml_trainer = MLModelTrainer()
    risk_manager = RiskManager()
    
    # Load existing ML models
    ml_trainer.load_models()
    
    # Start background tasks
    asyncio.create_task(update_market_data())
    asyncio.create_task(run_trading_cycle())
    asyncio.create_task(update_dashboard_data())
    
    yield
    
    # Shutdown
    logger.info("Shutting down trading bot dashboard...")

# Create FastAPI app
app = FastAPI(
    title="Trading Bot Dashboard",
    description="Advanced Trading Bot with ML and Risk Management",
    version="1.0.0",
    lifespan=lifespan
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# API Routes
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page"""
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/health-enhanced")
async def health_check():
    """Health check endpoint for enhanced trading bot"""
    return {
        "status": "healthy",
        "service": "trading-bot-enhanced",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/enhanced/account")
async def get_account():
    """Get account information"""
    try:
        if not trading_engine:
            raise HTTPException(status_code=500, detail="Trading engine not initialized")
        
        account_summary = trading_engine.get_account_summary()
        return account_summary
    except Exception as e:
        logger.error(f"Error getting account info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/enhanced/positions")
async def get_positions():
    """Get current positions with updated prices"""
    try:
        if not trading_engine:
            raise HTTPException(status_code=500, detail="Trading engine not initialized")
        
        # Update position prices with current market data
        updated_positions = []
        for pos in trading_engine.account.positions:
            try:
                # Get current price from market data
                current_price = trading_engine.market_data_collector.get_current_price(pos.symbol)
                if current_price:
                    # Calculate current P&L
                    if pos.side == 'long':
                        current_pnl = (current_price - pos.entry_price) * pos.size
                    else:  # short
                        current_pnl = (pos.entry_price - current_price) * pos.size
                    
                    # Calculate P&L percentage
                    pnl_percentage = (current_pnl / (pos.entry_price * pos.size)) * 100
                    
                    # Calculate duration with timezone awareness
                    duration_str = None
                    timestamp_str = None
                    if hasattr(pos, 'timestamp') and pos.timestamp:
                        # Convert to UTC+3 for display (VPS is UTC, browser is UTC+3)
                        utc_timestamp = pos.timestamp
                        utc_plus_3 = utc_timestamp + timedelta(hours=3)
                        timestamp_str = utc_plus_3.strftime("%Y-%m-%d %H:%M:%S UTC+3")
                        
                        # Calculate duration
                        now_utc = datetime.now()
                        duration = now_utc - utc_timestamp
                        hours, remainder = divmod(duration.total_seconds(), 3600)
                        minutes, seconds = divmod(remainder, 60)
                        duration_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
                    
                    # Create updated position dict
                    pos_dict = {
                        'symbol': pos.symbol,
                        'side': pos.side,
                        'size': pos.size,
                        'entry_price': pos.entry_price,
                        'current_price': current_price,
                        'pnl': current_pnl,
                        'pnl_percentage': pnl_percentage,
                        'timestamp': timestamp_str,
                        'duration': duration_str,
                        'open_time_utc_plus_3': timestamp_str
                    }
                else:
                    # Fallback to original position data if price unavailable
                    pos_dict = pos.__dict__.copy()
                    if hasattr(pos, 'timestamp'):
                        pos_dict['timestamp'] = pos.timestamp.isoformat()
                        pos_dict['duration'] = str(datetime.now() - pos.timestamp)
                
                updated_positions.append(pos_dict)
            except Exception as e:
                logger.error(f"Error updating position {pos.symbol}: {e}")
                # Fallback to original position data
                pos_dict = pos.__dict__.copy()
                if hasattr(pos, 'timestamp'):
                    pos_dict['timestamp'] = pos.timestamp.isoformat()
                    pos_dict['duration'] = str(datetime.now() - pos.timestamp)
                updated_positions.append(pos_dict)
        
        return {"positions": updated_positions}
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/enhanced/orders")
async def get_orders():
    """Get recent orders with enhanced information"""
    try:
        if not trading_engine:
            raise HTTPException(status_code=500, detail="Trading engine not initialized")
        
        # Get recent orders and enhance with PnL and duration info
        enhanced_orders = []
        for order in trading_engine.account.orders[-20:]:
            order_dict = order.__dict__.copy()
            
            # Add timestamp formatting
            if hasattr(order, 'timestamp'):
                order_dict['timestamp'] = order.timestamp.isoformat()
                order_dict['duration'] = str(datetime.now() - order.timestamp)
            
            # Add PnL information if available
            if hasattr(order, 'pnl'):
                order_dict['pnl'] = order.pnl
                order_dict['pnl_percentage'] = (order.pnl / (order.price * order.size)) * 100 if order.price and order.size else 0
            
            enhanced_orders.append(order_dict)
        
        return {"orders": enhanced_orders}
    except Exception as e:
        logger.error(f"Error getting orders: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/enhanced/market-data")
async def get_market_data():
    """Get market data summary"""
    try:
        if not market_data_collector:
            raise HTTPException(status_code=500, detail="Market data collector not initialized")
        
        symbols = settings.trading_pairs[:10]  # Limit to first 10 symbols
        market_summary = await market_data_collector.get_market_summary(symbols)
        return market_summary
    except Exception as e:
        logger.error(f"Error getting market data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/enhanced/performance")
async def get_performance():
    """Get performance metrics"""
    try:
        if not trading_engine:
            raise HTTPException(status_code=500, detail="Trading engine not initialized")
        
        performance = trading_engine.get_performance_metrics()
        return performance
    except Exception as e:
        logger.error(f"Error getting performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Pairs management
_PAIRS_FILE = Path(os.environ.get("PAIRS_FILE", "/app/config_pairs.json"))

def _read_pairs():
    try:
        if _PAIRS_FILE.exists():
            return json.loads(_PAIRS_FILE.read_text(encoding="utf-8")).get("pairs", [])
    except Exception:
        pass
    # fallback defaults
    return ["BTC/USDT","ETH/USDT","SOL/USDT","AVAX/USDT","LINK/USDT","UNI/USDT","DOT/USDT"]

def _write_pairs(pairs):
    try:
        _PAIRS_FILE.parent.mkdir(parents=True, exist_ok=True)
        _PAIRS_FILE.write_text(json.dumps({"pairs": pairs}, indent=2), encoding="utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to persist pairs: {e}")

class PairItem(BaseModel):
    symbol: str

@app.get("/api/enhanced/pairs")
def get_pairs():
    return {"pairs": _read_pairs()}

@app.post("/api/enhanced/pairs")
def add_pair(item: PairItem):
    pairs = _read_pairs()
    sym = item.symbol.strip().upper()
    if sym not in pairs:
        pairs.append(sym)
        _write_pairs(pairs)
    return {"pairs": pairs}

@app.delete("/api/enhanced/pairs")
def delete_pair(item: PairItem):
    pairs = _read_pairs()
    sym = item.symbol.strip().upper()
    pairs = [p for p in pairs if p != sym]
    _write_pairs(pairs)
    return {"pairs": pairs}

@app.get("/api/enhanced/ml-models")
async def get_ml_models():
    """Get ML model information"""
    try:
        if not ml_trainer:
            raise HTTPException(status_code=500, detail="ML trainer not initialized")
        
        model_performance = ml_trainer.get_model_performance()
        return model_performance
    except Exception as e:
        logger.error(f"Error getting ML models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/enhanced/trading-config")
async def update_trading_config(config: TradingConfig):
    """Update trading configuration"""
    try:
        # Update settings
        settings.trading_mode = config.trading_mode
        settings.risk_level = config.risk_level
        settings.max_positions = config.max_positions
        settings.max_daily_loss = config.max_daily_loss
        settings.max_position_size = config.max_position_size
        
        # Update risk manager
        if risk_manager:
            risk_manager.risk_params = risk_manager.RISK_PARAMETERS[config.risk_level]
        
        return {"message": "Configuration updated successfully"}
    except Exception as e:
        logger.error(f"Error updating trading config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/enhanced/close-position/{symbol}")
async def close_position(symbol: str):
    """Close a specific position"""
    try:
        if not trading_engine:
            raise HTTPException(status_code=500, detail="Trading engine not initialized")
        
        # Find position
        position = None
        for pos in trading_engine.account.positions:
            if pos.symbol == symbol:
                position = pos
                break
        
        if not position:
            raise HTTPException(status_code=404, detail="Position not found")
        
        # Close position
        await trading_engine.close_position(position, "manual")
        
        return {"message": f"Position {symbol} closed successfully"}
    except Exception as e:
        logger.error(f"Error closing position: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/enhanced/retrain-models")
async def retrain_models():
    """Retrain ML models"""
    try:
        if not ml_trainer or not trading_engine:
            raise HTTPException(status_code=500, detail="Components not initialized")
        
        # Get historical data
        feature_data = trading_engine.feature_data
        trade_outcomes = trading_engine.trade_outcomes
        
        if not feature_data or not trade_outcomes:
            raise HTTPException(status_code=400, detail="Insufficient data for training")
        
        # Prepare training data
        X, y = ml_trainer.prepare_training_data(feature_data, trade_outcomes)
        
        if X.empty or y.empty:
            raise HTTPException(status_code=400, detail="No valid training data")
        
        # Train models
        trained_models = ml_trainer.train_models(X, y)
        
        # Train LSTM model
        lstm_model = ml_trainer.train_lstm_model(X, y)
        if lstm_model:
            trained_models['lstm'] = lstm_model
        
        # Update global models
        ml_trainer.models.update(trained_models)
        
        return {
            "message": "Models retrained successfully",
            "trained_models": list(trained_models.keys()),
            "training_samples": len(X)
        }
    except Exception as e:
        logger.error(f"Error retraining models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Background tasks
async def update_market_data():
    """Update market data periodically"""
    while True:
        try:
            if market_data_collector:
                # Update market data for all trading pairs
                for symbol in settings.trading_pairs:
                    await market_data_collector.get_ohlcv_data(symbol, "15m", 100)
            
            await asyncio.sleep(60)  # Update every minute
        except Exception as e:
            logger.error(f"Error updating market data: {e}")
            await asyncio.sleep(60)

async def run_trading_cycle():
    """Run trading strategy cycle"""
    while True:
        try:
            if trading_engine and market_data_collector:
                await trading_engine.run_strategy_cycle(settings.trading_pairs)
            
            await asyncio.sleep(30)  # Run every 30 seconds
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            await asyncio.sleep(30)

async def update_dashboard_data():
    """Send real-time updates to dashboard"""
    while True:
        try:
            if trading_engine and manager.active_connections:
                # Get latest data
                account_data = trading_engine.get_account_summary()
                
                # Send update to all connected clients
                await manager.broadcast(json.dumps({
                    "type": "account_update",
                    "data": account_data,
                    "timestamp": datetime.now().isoformat()
                }))
            
            await asyncio.sleep(5)  # Update every 5 seconds
        except Exception as e:
            logger.error(f"Error updating dashboard: {e}")
            await asyncio.sleep(5)

if __name__ == "__main__":
    uvicorn.run(
        "dashboard:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
