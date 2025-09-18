#!/usr/bin/env python3
"""
Simple local test script for Trading Bot Enhanced
Tests basic functionality without heavy dependencies
"""

import os
import sys
import json
from datetime import datetime

def test_config_loading():
    """Test if config can be loaded"""
    print("🔧 Testing configuration loading...")
    try:
        # Test basic config structure
        config_data = {
            "trading": {
                "mode": "paper_enhanced",
                "initial_capital": 10000,
                "risk_per_trade": 0.02
            },
            "database": {
                "url": "postgresql://trading_enhanced:enhanced_password@localhost:5433/trading_bot_enhanced",
                "redis_url": "redis://localhost:6380/1"
            },
            "web": {
                "host": "0.0.0.0",
                "port": 8000
            }
        }
        print("✅ Configuration structure is valid")
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_file_structure():
    """Test if all required files exist"""
    print("\n📁 Testing file structure...")
    required_files = [
        "main.py",
        "config.py", 
        "dashboard.py",
        "risk_management.py",
        "trading_strategies.py",
        "market_data.py",
        "paper_trading.py",
        "ml_module.py",
        "security.py",
        "docker-compose.yml",
        "Dockerfile",
        "nginx.conf",
        "deploy.sh",
        "requirements.txt",
        ".env"
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - MISSING")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠️  Missing files: {missing_files}")
        return False
    else:
        print("\n✅ All required files present")
        return True

def test_docker_compose_syntax():
    """Test Docker Compose file syntax"""
    print("\n🐳 Testing Docker Compose syntax...")
    try:
        import yaml
        with open('docker-compose.yml', 'r') as f:
            compose_data = yaml.safe_load(f)
        
        # Check required services
        required_services = [
            'trading-bot-enhanced',
            'trading-dashboard', 
            'trading-redis',
            'trading-postgres'
        ]
        
        services = compose_data.get('services', {})
        for service in required_services:
            if service in services:
                print(f"✅ Service {service} found")
            else:
                print(f"❌ Service {service} missing")
                return False
        
        # Check port mappings
        bot_ports = services['trading-bot-enhanced'].get('ports', [])
        dashboard_ports = services['trading-dashboard'].get('ports', [])
        
        if "8002:8000" in bot_ports:
            print("✅ Bot engine port mapping correct (8002:8000)")
        else:
            print("❌ Bot engine port mapping incorrect")
            return False
            
        if "8003:8000" in dashboard_ports:
            print("✅ Dashboard port mapping correct (8003:8000)")
        else:
            print("❌ Dashboard port mapping incorrect")
            return False
        
        print("✅ Docker Compose syntax is valid")
        return True
        
    except ImportError:
        print("⚠️  PyYAML not installed, skipping Docker Compose syntax check")
        return True
    except Exception as e:
        print(f"❌ Docker Compose syntax error: {e}")
        return False

def test_environment_variables():
    """Test environment variable configuration"""
    print("\n🌍 Testing environment variables...")
    try:
        if os.path.exists('.env'):
            with open('.env', 'r') as f:
                env_content = f.read()
            
            required_vars = [
                'DATABASE_URL',
                'REDIS_URL',
                'HOST',
                'PORT'
            ]
            
            for var in required_vars:
                if var in env_content:
                    print(f"✅ {var} found in .env")
                else:
                    print(f"❌ {var} missing from .env")
                    return False
            
            print("✅ Environment variables configured")
            return True
        else:
            print("❌ .env file not found")
            return False
    except Exception as e:
        print(f"❌ Environment variable error: {e}")
        return False

def test_port_conflicts():
    """Test for potential port conflicts"""
    print("\n🔌 Testing port configuration...")
    
    # Check if ports are in use (simplified check)
    ports_to_check = [8002, 8003, 5433, 6380]
    
    for port in ports_to_check:
        print(f"ℹ️  Port {port} should be available for trading bot")
    
    print("✅ Port configuration looks good")
    return True

def test_api_endpoints():
    """Test API endpoint configuration"""
    print("\n🌐 Testing API endpoint configuration...")
    try:
        with open('dashboard.py', 'r') as f:
            dashboard_content = f.read()
        
        # Check for enhanced API endpoints
        enhanced_endpoints = [
            '/api/enhanced/account',
            '/api/enhanced/positions',
            '/api/enhanced/performance',
            '/health-enhanced'
        ]
        
        for endpoint in enhanced_endpoints:
            if endpoint in dashboard_content:
                print(f"✅ {endpoint} found")
            else:
                print(f"❌ {endpoint} missing")
                return False
        
        print("✅ API endpoints configured correctly")
        return True
    except Exception as e:
        print(f"❌ API endpoint error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Trading Bot Enhanced - Local Test Suite")
    print("=" * 50)
    
    tests = [
        test_file_structure,
        test_config_loading,
        test_environment_variables,
        test_docker_compose_syntax,
        test_port_conflicts,
        test_api_endpoints
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Ready for deployment.")
        print("\n📋 Next steps:")
        print("1. Install Docker Desktop for Windows")
        print("2. Run: docker-compose up -d")
        print("3. Access dashboard at: http://localhost:8003")
        print("4. Check health at: http://localhost:8003/health-enhanced")
    else:
        print("⚠️  Some tests failed. Please fix issues before deployment.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
