#!/usr/bin/env python3
"""
Test script to verify trading bot installation
"""
import sys
import importlib
import traceback
from datetime import datetime

def test_import(module_name, description):
    """Test if a module can be imported"""
    try:
        importlib.import_module(module_name)
        print(f"✅ {description}")
        return True
    except ImportError as e:
        print(f"❌ {description} - {e}")
        return False
    except Exception as e:
        print(f"⚠️  {description} - {e}")
        return False

def test_config():
    """Test configuration loading"""
    try:
        from config import settings
        print(f"✅ Configuration loaded - Trading mode: {settings.trading_mode}")
        return True
    except Exception as e:
        print(f"❌ Configuration error - {e}")
        return False

def test_components():
    """Test core components"""
    try:
        from risk_management import RiskManager
        from trading_strategies import StrategyEnsemble
        from market_data import MarketDataCollector
        from paper_trading import PaperTradingEngine
        from ml_module import MLModelTrainer
        
        print("✅ Core components imported successfully")
        return True
    except Exception as e:
        print(f"❌ Core components error - {e}")
        return False

def test_database_connection():
    """Test database connection"""
    try:
        import sqlite3
        conn = sqlite3.connect(':memory:')
        conn.close()
        print("✅ Database connection test passed")
        return True
    except Exception as e:
        print(f"❌ Database connection error - {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("TRADING BOT INSTALLATION TEST")
    print("=" * 60)
    print(f"Test started at: {datetime.now()}")
    print(f"Python version: {sys.version}")
    print()
    
    tests = [
        ("Core Python modules", [
            ("pandas", "Pandas data analysis"),
            ("numpy", "NumPy numerical computing"),
            ("asyncio", "AsyncIO for async programming"),
            ("json", "JSON handling"),
            ("datetime", "Date/time handling"),
        ]),
        ("Trading libraries", [
            ("ccxt", "Cryptocurrency exchange library"),
            ("ta", "Technical analysis library"),
        ]),
        ("Web framework", [
            ("fastapi", "FastAPI web framework"),
            ("uvicorn", "ASGI server"),
            ("websockets", "WebSocket support"),
        ]),
        ("Machine learning", [
            ("sklearn", "Scikit-learn ML library"),
            ("tensorflow", "TensorFlow deep learning"),
            ("xgboost", "XGBoost gradient boosting"),
        ]),
        ("Database", [
            ("sqlalchemy", "SQLAlchemy ORM"),
            ("redis", "Redis client"),
        ]),
        ("Visualization", [
            ("plotly", "Plotly charts"),
            ("dash", "Dash web apps"),
        ]),
    ]
    
    passed = 0
    total = 0
    
    for category, modules in tests:
        print(f"\n{category}:")
        print("-" * 40)
        
        for module, description in modules:
            total += 1
            if test_import(module, description):
                passed += 1
    
    print(f"\nConfiguration:")
    print("-" * 40)
    total += 1
    if test_config():
        passed += 1
    
    print(f"\nCore Components:")
    print("-" * 40)
    total += 1
    if test_components():
        passed += 1
    
    print(f"\nDatabase:")
    print("-" * 40)
    total += 1
    if test_database_connection():
        passed += 1
    
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Passed: {passed}/{total} tests")
    print(f"Success rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n🎉 All tests passed! The trading bot is ready to use.")
        print("\nNext steps:")
        print("1. Edit .env file with your configuration")
        print("2. Run: python main.py --mode bot")
        print("3. Open dashboard at: http://localhost:8000")
        print("   Note: If running alongside Claude Bot, use port 8003 for dashboard")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please install missing dependencies.")
        print("\nTo install all dependencies, run:")
        print("pip install -r requirements.txt")
    
    print("\n" + "=" * 60)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
