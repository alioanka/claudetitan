# Trading Bot Enhanced - Advanced Automated Trading System

A highly sophisticated, fully automated trading bot with excellent risk management, machine learning integration, and comprehensive monitoring capabilities.

## 🚀 Features

### Core Trading Features
- **Fully Automated Trading**: 24/7 operation with minimal human intervention
- **Advanced Risk Management**: Kelly Criterion, stop-loss, take-profit, position sizing
- **Multiple Trading Strategies**: Scalping, momentum, mean reversion, arbitrage
- **Paper Trading Mode**: Safe testing and data collection before live trading
- **Machine Learning Integration**: Self-improving strategies using ML models

### Risk Management
- **Position Sizing**: Dynamic position sizing based on account balance and risk tolerance
- **Stop Loss & Take Profit**: Automated risk controls for every trade
- **Daily Loss Limits**: Configurable maximum daily loss limits
- **Portfolio Diversification**: Automatic spread across multiple trading pairs
- **Real-time Risk Monitoring**: Continuous risk assessment and alerts

### Machine Learning
- **Multiple ML Models**: Random Forest, XGBoost, LSTM, SVM, Logistic Regression
- **Feature Engineering**: Advanced technical indicators and market features
- **Auto-Retraining**: Models automatically retrain with new data
- **Performance Monitoring**: Real-time model performance tracking
- **Strategy Optimization**: ML-driven strategy parameter optimization

### Monitoring & Dashboard
- **Real-time Dashboard**: Web-based monitoring interface
- **Mobile Responsive**: Access from any device
- **Performance Metrics**: P&L, win rate, Sharpe ratio, drawdown analysis
- **Trade History**: Complete trade log with detailed analytics
- **Alert System**: Email, SMS, and webhook notifications

### Infrastructure
- **Docker Deployment**: Easy deployment with Docker Compose
- **VPS Ready**: Optimized for 24/7 VPS operation
- **Database Integration**: PostgreSQL for data persistence
- **Redis Caching**: High-performance caching layer
- **Nginx Reverse Proxy**: Production-ready web server
- **SSL Support**: Secure HTTPS connections
- **Monitoring Stack**: Prometheus + Grafana integration

## 📋 Quick Start Guide

### Step 1: Clone from GitHub
```bash
git clone https://github.com/alioanka/claudetitan.git
cd claudetitan
```

### Step 2: Local Testing (Recommended First)
```bash
# Install Docker Desktop for Windows
# Download from: https://www.docker.com/products/docker-desktop/

# Test locally
python test_local_simple.py

# Start with Docker
docker-compose up -d

# Access dashboard
open http://localhost:8003
```

### Step 3: Deploy to VPS
```bash
# On your VPS
git clone https://github.com/alioanka/claudetitan.git
cd claudetitan

# Configure environment
cp env_example.txt .env
nano .env  # Edit with your settings

# Deploy
chmod +x deploy.sh
./deploy.sh
```

## 🛠️ Complete Installation Guide

### Prerequisites
- **Local Testing**: Docker Desktop for Windows
- **VPS Deployment**: Ubuntu 20.04+ with Docker installed
- **Resources**: 4GB+ RAM, 20GB+ storage
- **Ports**: 8002, 8003, 5433, 6380 available

### Local Development Setup

#### 1. Install Docker Desktop
- Download from: https://www.docker.com/products/docker-desktop/
- Install and restart your computer
- Verify: `docker --version`

#### 2. Clone and Test
```bash
# Clone repository
git clone https://github.com/alioanka/claudetitan.git
cd claudetitan

# Run tests
python test_local_simple.py

# Start services
docker-compose up -d

# Check status
docker-compose ps
docker-compose logs -f
```

#### 3. Access Dashboard
- **Dashboard**: http://localhost:8003
- **Health Check**: http://localhost:8003/health-enhanced
- **API Docs**: http://localhost:8003/docs

### VPS Deployment Guide

#### 1. Prepare Your VPS
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Logout and login to apply docker group changes
```

#### 2. Clone and Configure
```bash
# Clone repository
git clone https://github.com/alioanka/claudetitan.git
cd claudetitan

# Create environment file
cp env_example.txt .env

# Edit configuration
nano .env
```

#### 3. Configure Environment Variables
```bash
# Edit .env file with your settings
nano .env

# Key settings to configure:
# - DATABASE_URL (if using external database)
# - REDIS_URL (if using external Redis)
# - BINANCE_API_KEY (for live trading)
# - BINANCE_SECRET_KEY (for live trading)
# - EMAIL settings (for notifications)
```

#### 4. Deploy with Docker
```bash
# Make deploy script executable
chmod +x deploy.sh

# Run deployment
./deploy.sh

# Or manually start
docker-compose up -d
```

#### 5. Verify Deployment
```bash
# Check running containers
docker-compose ps

# Check logs
docker-compose logs -f

# Test endpoints
curl http://localhost:8003/health-enhanced
curl http://localhost:8003/api/enhanced/account
```

#### 6. Configure Firewall (if needed)
```bash
# Allow required ports
sudo ufw allow 8003  # Dashboard
sudo ufw allow 8002  # Bot engine
sudo ufw allow 5433  # PostgreSQL
sudo ufw allow 6380  # Redis
sudo ufw allow 9091  # Prometheus
sudo ufw allow 3001  # Grafana
sudo ufw allow 8080  # Nginx HTTP
sudo ufw allow 8443  # Nginx HTTPS

# Enable firewall
sudo ufw enable
```

## ⚙️ Configuration

### Environment Variables
```bash
# Trading Configuration
TRADING_MODE=paper_enhanced  # paper_enhanced, live
RISK_LEVEL=moderate          # conservative, moderate, aggressive
INITIAL_CAPITAL=10000

# Database Configuration
DATABASE_URL=postgresql://trading_enhanced:enhanced_password@localhost:5433/trading_bot_enhanced
REDIS_URL=redis://localhost:6380/1

# Web Dashboard
HOST=0.0.0.0
PORT=8000
DEBUG=false

# API Keys (for live trading)
BINANCE_API_KEY=your_api_key_here
BINANCE_SECRET_KEY=your_secret_key_here

# Notifications
EMAIL_SMTP_SERVER=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
```

### Trading Parameters
```python
# Risk Management
MAX_DAILY_LOSS = 0.05    # 5% of account
RISK_PER_TRADE = 0.02    # 2% per trade
MAX_POSITIONS = 10       # Maximum concurrent positions

# Trading Pairs
TRADING_PAIRS = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT']

# Strategy Parameters
SCALP_TIMEFRAME = '1m'
MOMENTUM_TIMEFRAME = '5m'
ARBITRAGE_THRESHOLD = 0.001  # 0.1%
```

## 📊 Usage

### Starting the Bot
```bash
# Paper trading mode (recommended for testing)
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f trading-bot-enhanced
```

### Dashboard Access
- **Local**: http://localhost:8003
- **VPS**: http://your-vps-ip:8003
- **Mobile**: Responsive design works on all devices

### API Endpoints
```bash
# Health check
GET /health-enhanced

# Account information
GET /api/enhanced/account

# Current positions
GET /api/enhanced/positions

# Performance metrics
GET /api/enhanced/performance

# Trading configuration
POST /api/enhanced/trading-config
```

## 🔧 Development

### Project Structure
```
claudetitan/
├── main.py                 # Main entry point
├── config.py              # Configuration management
├── risk_management.py     # Risk management system
├── trading_strategies.py  # Trading strategies
├── market_data.py         # Market data collection
├── paper_trading.py       # Paper trading system
├── ml_module.py          # Machine learning module
├── dashboard.py           # Web dashboard
├── security.py           # Security and authentication
├── templates/            # HTML templates
├── static/              # Static assets
├── docker-compose.yml   # Docker configuration
├── Dockerfile          # Docker image
├── deploy.sh           # Deployment script
├── test_local_simple.py # Local testing script
└── requirements.txt    # Python dependencies
```

### Running Tests
```bash
# Run local tests
python test_local_simple.py

# Test Docker build
docker-compose build

# Test services
docker-compose up -d
docker-compose ps
```

## 🚀 Deployment Commands

### Push to GitHub
```bash
# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit changes
git commit -m "Initial commit: Trading Bot Enhanced"

# Add remote origin
git remote add origin https://github.com/alioanka/claudetitan.git

# Push to GitHub
git push -u origin main
```

### Deploy to VPS
```bash
# On your VPS
git clone https://github.com/alioanka/claudetitan.git
cd claudetitan

# Configure and deploy
cp env_example.txt .env
nano .env  # Edit configuration
chmod +x deploy.sh
./deploy.sh
```

## 📈 Monitoring

### Performance Metrics
- **Total P&L**: Overall profit/loss
- **Win Rate**: Percentage of profitable trades
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Average Trade Duration**: Mean time per trade

### Service Monitoring
```bash
# Check container status
docker-compose ps

# View logs
docker-compose logs -f

# Check resource usage
docker stats

# Restart services
docker-compose restart
```

## 🔒 Security

### API Security
- **JWT Authentication**: Secure API access
- **Rate Limiting**: Prevent API abuse
- **Input Validation**: Sanitize all inputs
- **HTTPS Only**: Encrypted communications

### Trading Security
- **API Key Encryption**: Encrypted storage of exchange keys
- **Withdrawal Protection**: No automatic withdrawals
- **Position Limits**: Maximum position size limits
- **Audit Logging**: Complete audit trail

## 🆘 Troubleshooting

### Common Issues

#### Docker Issues
```bash
# Check Docker status
docker --version
docker-compose --version

# Restart Docker service
sudo systemctl restart docker

# Clean up containers
docker-compose down
docker system prune -a
```

#### Port Conflicts
```bash
# Check port usage
netstat -tulpn | grep :8003
netstat -tulpn | grep :5433

# Kill processes using ports
sudo fuser -k 8003/tcp
sudo fuser -k 5433/tcp
```

#### Database Issues
```bash
# Check database logs
docker-compose logs trading-postgres

# Reset database
docker-compose down -v
docker-compose up -d
```

## 📞 Support

- **GitHub Repository**: https://github.com/alioanka/claudetitan
- **Issues**: https://github.com/alioanka/claudetitan/issues
- **Documentation**: See files in repository

## ⚠️ Disclaimer

This software is for educational and research purposes only. Trading cryptocurrencies involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results. Use at your own risk.

## 🙏 Acknowledgments

- [CCXT](https://github.com/ccxt/ccxt) - Cryptocurrency exchange trading library
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [Pandas](https://pandas.pydata.org/) - Data analysis library
- [Scikit-learn](https://scikit-learn.org/) - Machine learning library