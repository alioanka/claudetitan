#!/usr/bin/env python3
"""
Create logs directory structure for Trading Bot
"""
import os
from pathlib import Path

def create_logs_structure():
    """Create the logs directory structure"""
    logs_dir = Path("logs")
    
    # Create main logs directory
    logs_dir.mkdir(exist_ok=True)
    
    # Create subdirectories for different log types
    subdirs = [
        "trading",
        "risk",
        "market_data", 
        "ml",
        "dashboard",
        "errors",
        "debug"
    ]
    
    for subdir in subdirs:
        (logs_dir / subdir).mkdir(exist_ok=True)
    
    # Create .gitkeep files to ensure directories are tracked
    for subdir in subdirs:
        (logs_dir / subdir / ".gitkeep").touch()
    
    print("✅ Logs directory structure created:")
    print(f"📁 {logs_dir.absolute()}")
    for subdir in subdirs:
        print(f"  📁 {subdir}/")
    
    print("\n📋 Log files will be created in:")
    print("  📄 trading_bot_YYYYMMDD.log - Main application logs")
    print("  📄 trading_bot_errors_YYYYMMDD.log - Error logs only")
    print("  📄 trading_bot_debug_YYYYMMDD.log - Debug logs (if DEBUG mode)")
    print("  📄 strategies_YYYYMMDD.log - Trading strategies logs")
    print("  📄 risk_management_YYYYMMDD.log - Risk management logs")
    print("  📄 market_data_YYYYMMDD.log - Market data logs")
    print("  📄 ml_module_YYYYMMDD.log - ML module logs")
    print("  📄 dashboard_YYYYMMDD.log - Dashboard logs")

if __name__ == "__main__":
    create_logs_structure()
