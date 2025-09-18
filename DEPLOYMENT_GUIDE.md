# Complete Deployment Guide - Trading Bot Enhanced

## 🚀 Step-by-Step Deployment Instructions

### Phase 1: Push to GitHub

#### Step 1: Initialize Git Repository
```bash
# Navigate to your project directory
cd C:\Users\HP\Desktop\ClaudeTitan

# Initialize git repository
git init

# Add all files to staging
git add .

# Create initial commit
git commit -m "Initial commit: Trading Bot Enhanced with complete conflict resolution"
```

#### Step 2: Connect to GitHub Repository
```bash
# Add remote origin
git remote add origin https://github.com/alioanka/claudetitan.git

# Verify remote connection
git remote -v
```

#### Step 3: Push to GitHub
```bash
# Push to main branch
git push -u origin main

# If you get authentication error, use:
# git push -u origin main --force
```

### Phase 2: Local Testing (Optional but Recommended)

#### Step 1: Install Docker Desktop
1. Download Docker Desktop from: https://www.docker.com/products/docker-desktop/
2. Install and restart your computer
3. Verify installation: `docker --version`

#### Step 2: Test Locally
```bash
# Run local tests
python test_local_simple.py

# Start services
docker-compose up -d

# Check status
docker-compose ps

# Access dashboard
# Open browser: http://localhost:8003
```

#### Step 3: Stop Local Testing
```bash
# Stop services
docker-compose down
```

### Phase 3: VPS Deployment

#### Step 1: Connect to Your VPS
```bash
# SSH into your VPS
ssh username@your-vps-ip

# Or use your preferred SSH client
```

#### Step 2: Install Docker on VPS
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

# Verify installations
docker --version
docker-compose --version

# Logout and login to apply docker group changes
exit
ssh username@your-vps-ip
```

#### Step 3: Clone Repository
```bash
# Clone from GitHub
git clone https://github.com/alioanka/claudetitan.git
cd claudetitan

# Verify files
ls -la
```

#### Step 4: Configure Environment
```bash
# Copy environment template
cp env_example.txt .env

# Edit configuration
nano .env

# Key settings to configure:
# - BINANCE_API_KEY (for live trading)
# - BINANCE_SECRET_KEY (for live trading)
# - EMAIL settings (for notifications)
# - Any other custom settings
```

#### Step 5: Deploy with Docker
```bash
# Make deploy script executable
chmod +x deploy.sh

# Run deployment
./deploy.sh

# Or manually start
docker-compose up -d
```

#### Step 6: Verify Deployment
```bash
# Check running containers
docker-compose ps

# Check logs
docker-compose logs -f

# Test endpoints
curl http://localhost:8003/health-enhanced
curl http://localhost:8003/api/enhanced/account
```

#### Step 7: Configure Firewall (if needed)
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

### Phase 4: Access and Monitor

#### Step 1: Access Dashboard
- **Dashboard**: http://your-vps-ip:8003
- **Health Check**: http://your-vps-ip:8003/health-enhanced
- **API Documentation**: http://your-vps-ip:8003/docs

#### Step 2: Monitor Services
```bash
# Check container status
docker-compose ps

# View real-time logs
docker-compose logs -f

# Check resource usage
docker stats

# Restart services if needed
docker-compose restart
```

#### Step 3: Set Up Monitoring (Optional)
```bash
# Access Prometheus
# http://your-vps-ip:9091

# Access Grafana
# http://your-vps-ip:3001
# Default login: admin/admin
```

## 🔧 Configuration Details

### Environment Variables (.env)
```bash
# Trading Configuration
TRADING_MODE=paper_enhanced
RISK_LEVEL=moderate
INITIAL_CAPITAL=10000

# Database Configuration (Docker handles this)
DATABASE_URL=postgresql://trading_enhanced:enhanced_password@trading-postgres:5433/trading_bot_enhanced
REDIS_URL=redis://trading-redis:6380/1

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

### Port Configuration
| Service | Internal Port | External Port | Access URL |
|---------|---------------|---------------|------------|
| **Dashboard** | 8000 | 8003 | http://your-vps-ip:8003 |
| **Bot Engine** | 8000 | 8002 | http://your-vps-ip:8002 |
| **PostgreSQL** | 5433 | 5433 | localhost:5433 |
| **Redis** | 6380 | 6380 | localhost:6380 |
| **Prometheus** | 9090 | 9091 | http://your-vps-ip:9091 |
| **Grafana** | 3000 | 3001 | http://your-vps-ip:3001 |
| **Nginx HTTP** | 80 | 8080 | http://your-vps-ip:8080 |
| **Nginx HTTPS** | 443 | 8443 | https://your-vps-ip:8443 |

## 🆘 Troubleshooting

### Common Issues and Solutions

#### 1. Git Push Issues
```bash
# If authentication fails
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# If remote already exists
git remote remove origin
git remote add origin https://github.com/alioanka/claudetitan.git

# Force push if needed
git push -u origin main --force
```

#### 2. Docker Issues
```bash
# Check Docker status
sudo systemctl status docker

# Restart Docker
sudo systemctl restart docker

# Check disk space
df -h

# Clean up Docker
docker system prune -a
```

#### 3. Port Conflicts
```bash
# Check port usage
sudo netstat -tulpn | grep :8003
sudo netstat -tulpn | grep :5433

# Kill processes using ports
sudo fuser -k 8003/tcp
sudo fuser -k 5433/tcp
```

#### 4. Database Issues
```bash
# Check database logs
docker-compose logs trading-postgres

# Reset database
docker-compose down -v
docker-compose up -d
```

#### 5. Permission Issues
```bash
# Fix file permissions
sudo chown -R $USER:$USER /path/to/claudetitan
chmod +x deploy.sh
```

## 📊 Verification Checklist

### Before Deployment
- [ ] All files committed to git
- [ ] Repository pushed to GitHub
- [ ] Docker installed on VPS
- [ ] Ports 8002, 8003, 5433, 6380 available
- [ ] Environment variables configured

### After Deployment
- [ ] All containers running (`docker-compose ps`)
- [ ] Dashboard accessible at http://your-vps-ip:8003
- [ ] Health check returns healthy status
- [ ] API endpoints responding
- [ ] No error messages in logs
- [ ] Database and Redis connections working

### Performance Check
- [ ] Dashboard loads quickly
- [ ] API responses under 1 second
- [ ] Memory usage under 4GB
- [ ] CPU usage reasonable
- [ ] No memory leaks

## 🔄 Updates and Maintenance

### Updating the Bot
```bash
# On VPS
cd claudetitan
git pull origin main
docker-compose down
docker-compose up -d
```

### Backup Data
```bash
# Backup database
docker-compose exec trading-postgres pg_dump -U trading_enhanced trading_bot_enhanced > backup.sql

# Backup Redis
docker-compose exec trading-redis redis-cli --rdb /data/backup.rdb
```

### Monitoring Commands
```bash
# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Check resources
docker stats

# Restart services
docker-compose restart
```

## ✅ Success Indicators

Your deployment is successful when:
1. ✅ All containers are running
2. ✅ Dashboard loads at http://your-vps-ip:8003
3. ✅ Health check returns `{"status": "healthy"}`
4. ✅ API endpoints respond correctly
5. ✅ No error messages in logs
6. ✅ Bot is running in paper trading mode
7. ✅ Database and Redis are connected

**Congratulations! Your Trading Bot Enhanced is now running on your VPS!** 🎉
