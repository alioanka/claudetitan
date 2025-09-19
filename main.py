"""
Main Trading Bot Orchestrator
"""
import asyncio
import logging
import signal
import sys
from datetime import datetime
from typing import Dict, List
import uvicorn
from contextlib import asynccontextmanager

from config import settings
from paper_trading import PaperTradingEngine
from market_data import MarketDataCollector
from ml_module import MLModelTrainer, MLStrategyOptimizer
from risk_management import RiskManager
from trading_strategies import StrategyEnsemble
from dashboard import app

# Configure simple logging
from simple_logging import setup_simple_logging, get_logger
setup_simple_logging(settings.log_level, "logs")

logger = logging.getLogger(__name__)

class TradingBot:
    """Main trading bot orchestrator"""
    
    def __init__(self):
        self.running = False
        self.trading_engine = None
        self.market_data_collector = None
        self.ml_trainer = None
        self.risk_manager = None
        self.strategy_ensemble = None
        self.tasks = []
        
        # Performance tracking
        self.start_time = None
        self.total_trades = 0
        self.successful_trades = 0
        
    async def initialize(self):
        """Initialize all components"""
        try:
            logger.info("Initializing trading bot...")
            
            # Initialize components
            self.trading_engine = PaperTradingEngine(initial_balance=10000.0)
            self.market_data_collector = MarketDataCollector()
            self.ml_trainer = MLModelTrainer()
            self.risk_manager = RiskManager()
            self.strategy_ensemble = StrategyEnsemble()
            
            # Load existing ML models
            self.ml_trainer.load_models()
            
            # Initialize market data collector
            self.market_data_collector.initialize_exchange()
            
            logger.info("Trading bot initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize trading bot: {e}")
            raise
    
    async def start(self):
        """Start the trading bot"""
        try:
            if self.running:
                logger.warning("Trading bot is already running")
                return
            
            logger.info("Starting trading bot...")
            self.running = True
            self.start_time = datetime.now()
            
            # Start background tasks
            self.tasks = [
                asyncio.create_task(self._market_data_loop()),
                asyncio.create_task(self._trading_loop()),
                asyncio.create_task(self._ml_training_loop()),
                asyncio.create_task(self._monitoring_loop()),
                asyncio.create_task(self._performance_tracking_loop())
            ]
            
            logger.info("Trading bot started successfully")
            
            # Wait for all tasks
            await asyncio.gather(*self.tasks)
            
        except Exception as e:
            logger.error(f"Error starting trading bot: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the trading bot"""
        try:
            if not self.running:
                logger.warning("Trading bot is not running")
                return
            
            logger.info("Stopping trading bot...")
            self.running = False
            
            # Cancel all tasks
            for task in self.tasks:
                task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*self.tasks, return_exceptions=True)
            
            # Close positions if in live mode
            if settings.trading_mode == "live" and self.trading_engine:
                await self._close_all_positions()
            
            logger.info("Trading bot stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping trading bot: {e}")
    
    async def _market_data_loop(self):
        """Market data collection loop"""
        while self.running:
            try:
                # Update market data for all trading pairs
                for symbol in settings.trading_pairs:
                    try:
                        await self.market_data_collector.get_ohlcv_data(symbol, settings.primary_timeframe, 100)
                    except Exception as e:
                        logger.error(f"Error updating market data for {symbol}: {e}")
                        continue
                
                # Wait before next update
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error in market data loop: {e}")
                await asyncio.sleep(60)
    
    async def _trading_loop(self):
        """Main trading loop"""
        while self.running:
            try:
                # Run trading strategy cycle
                await self.trading_engine.run_strategy_cycle(settings.trading_pairs)
                
                # Update performance metrics
                self._update_performance_metrics()
                
                # Wait before next cycle
                await asyncio.sleep(30)  # Run every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(30)
    
    async def _ml_training_loop(self):
        """ML model training loop"""
        while self.running:
            try:
                # Check if it's time to retrain models
                if self._should_retrain_models():
                    await self._retrain_models()
                
                # Wait before next check
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                logger.error(f"Error in ML training loop: {e}")
                await asyncio.sleep(3600)
    
    async def _monitoring_loop(self):
        """System monitoring loop"""
        while self.running:
            try:
                # Check system health
                await self._check_system_health()
                
                # Log performance metrics
                self._log_performance_metrics()
                
                # Wait before next check
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(300)
    
    async def _performance_tracking_loop(self):
        """Performance tracking loop"""
        while self.running:
            try:
                # Track performance metrics
                performance = self.trading_engine.get_performance_metrics()
                
                # Log key metrics
                if performance:
                    logger.info(f"Performance - Total Trades: {performance.get('total_trades', 0)}, "
                              f"Win Rate: {performance.get('win_rate', 0):.2%}, "
                              f"Total P&L: ${performance.get('total_pnl', 0):.2f}")
                
                # Wait before next update
                await asyncio.sleep(600)  # Update every 10 minutes
                
            except Exception as e:
                logger.error(f"Error in performance tracking loop: {e}")
                await asyncio.sleep(600)
    
    def _should_retrain_models(self) -> bool:
        """Check if models should be retrained"""
        try:
            if not self.ml_trainer.models:
                return True
            
            # Check if we have enough new data
            if len(self.trading_engine.feature_data) < 100:
                return False
            
            # Check if models are old
            for model in self.ml_trainer.models.values():
                if (datetime.now() - model.last_trained).total_seconds() > settings.model_retrain_interval * 3600:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking if should retrain models: {e}")
            return False
    
    async def _retrain_models(self):
        """Retrain ML models"""
        try:
            logger.info("Starting ML model retraining...")
            
            # Prepare training data
            X, y = self.ml_trainer.prepare_training_data(
                self.trading_engine.feature_data,
                self.trading_engine.trade_outcomes
            )
            
            if X.empty or y.empty:
                logger.warning("Insufficient data for model retraining")
                return
            
            # Train models
            trained_models = self.ml_trainer.train_models(X, y)
            
            # Train LSTM model
            lstm_model = self.ml_trainer.train_lstm_model(X, y)
            if lstm_model:
                trained_models['lstm'] = lstm_model
            
            # Update global models
            self.ml_trainer.models.update(trained_models)
            
            logger.info(f"Successfully retrained {len(trained_models)} models")
            
        except Exception as e:
            logger.error(f"Error retraining models: {e}")
    
    async def _check_system_health(self):
        """Check system health and performance"""
        try:
            # Check trading engine
            if not self.trading_engine:
                logger.error("Trading engine not initialized")
                return
            
            # Check market data collector
            if not self.market_data_collector:
                logger.error("Market data collector not initialized")
                return
            
            # Check account balance
            account_summary = self.trading_engine.get_account_summary()
            if account_summary['equity'] < 1000:  # Minimum balance threshold
                logger.warning(f"Low account balance: ${account_summary['equity']:.2f}")
            
            # Check risk metrics
            portfolio_risk = self.risk_manager.assess_portfolio_risk(
                [pos.__dict__ for pos in self.trading_engine.account.positions],
                self.trading_engine.account.equity
            )
            
            if portfolio_risk and portfolio_risk.risk_score > 80:
                logger.warning(f"High portfolio risk: {portfolio_risk.risk_score:.1f}")
            
            # Check ML models
            if not self.ml_trainer.models:
                logger.warning("No ML models available")
            
        except Exception as e:
            logger.error(f"Error checking system health: {e}")
    
    def _update_performance_metrics(self):
        """Update performance metrics"""
        try:
            performance = self.trading_engine.get_performance_metrics()
            if performance:
                self.total_trades = performance.get('total_trades', 0)
                self.successful_trades = int(performance.get('winning_trades', 0))
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    def _log_performance_metrics(self):
        """Log performance metrics"""
        try:
            if self.trading_engine:
                performance = self.trading_engine.get_performance_metrics()
                if performance:
                    logger.info(f"Performance Summary - "
                              f"Trades: {performance.get('total_trades', 0)}, "
                              f"Win Rate: {performance.get('win_rate', 0):.2%}, "
                              f"P&L: ${performance.get('total_pnl', 0):.2f}, "
                              f"Equity: ${self.trading_engine.account.equity:.2f}")
            
        except Exception as e:
            logger.error(f"Error logging performance metrics: {e}")
    
    def get_performance_metrics(self):
        """Get performance metrics for dashboard"""
        try:
            if not self.trading_engine:
                return {}
            
            # Get performance from trading engine
            performance = self.trading_engine.get_performance_metrics()
            
            # Add additional metrics
            if performance:
                performance.update({
                    'bot_status': 'running' if self.running else 'stopped',
                    'uptime': str(datetime.now() - self.start_time) if hasattr(self, 'start_time') else '0:00:00',
                    'last_update': datetime.now().isoformat()
                })
            
            return performance or {}
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}
    
    async def _close_all_positions(self):
        """Close all open positions"""
        try:
            if not self.trading_engine:
                return
            
            positions = self.trading_engine.account.positions.copy()
            for position in positions:
                await self.trading_engine.close_position(position, "shutdown")
            
            logger.info(f"Closed {len(positions)} positions")
            
        except Exception as e:
            logger.error(f"Error closing positions: {e}")
    
    def get_status(self) -> Dict:
        """Get bot status"""
        try:
            uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
            
            status = {
                'running': self.running,
                'uptime': uptime,
                'total_trades': self.total_trades,
                'successful_trades': self.successful_trades,
                'success_rate': self.successful_trades / max(self.total_trades, 1),
                'components': {
                    'trading_engine': self.trading_engine is not None,
                    'market_data_collector': self.market_data_collector is not None,
                    'ml_trainer': self.ml_trainer is not None,
                    'risk_manager': self.risk_manager is not None,
                    'strategy_ensemble': self.strategy_ensemble is not None
                }
            }
            
            if self.trading_engine:
                account_summary = self.trading_engine.get_account_summary()
                status['account'] = {
                    'balance': account_summary['balance'],
                    'equity': account_summary['equity'],
                    'total_pnl': account_summary['total_pnl'],
                    'daily_pnl': account_summary['daily_pnl'],
                    'positions': len(account_summary['positions'])
                }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting bot status: {e}")
            return {'error': str(e)}

# Global bot instance
bot = None

async def main():
    """Main function"""
    global bot
    
    try:
        # Create and initialize bot
        bot = TradingBot()
        await bot.initialize()
        
        # Set up signal handlers
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            asyncio.create_task(bot.stop())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start the bot
        await bot.start()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        if bot:
            await bot.stop()

def run_dashboard():
    """Run the dashboard server"""
    uvicorn.run(
        "dashboard:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced Trading Bot")
    parser.add_argument("--mode", choices=["bot", "dashboard"], default="bot",
                       help="Run mode: bot or dashboard")
    parser.add_argument("--host", default=settings.host,
                       help="Host for dashboard server")
    parser.add_argument("--port", type=int, default=settings.port,
                       help="Port for dashboard server")
    
    args = parser.parse_args()
    
    if args.mode == "dashboard":
        settings.host = args.host
        settings.port = args.port
        run_dashboard()
    else:
        asyncio.run(main())
