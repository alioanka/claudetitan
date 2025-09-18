# Bot Management Guide - Running Both Bots on Same VPS

## 🎯 **Overview**

You now have **two separate trading bots** running on the same VPS:
- **Claude Bot** (existing) - Port 8000
- **ClaudeTitan Bot** (new) - Port 8003

Both bots are completely isolated and can run simultaneously without conflicts.

## 📁 **Directory Structure**

```
/root/
├── ClaudeBot/           # Existing Claude Bot
│   ├── docker-compose.yml
│   ├── main.py
│   └── ...
└── claudetitan/         # New ClaudeTitan Bot
    ├── docker-compose.yml
    ├── main.py
    └── ...
```

## 🔄 **Update Commands**

### **Claude Bot (Existing)**
```bash
# Navigate to Claude Bot directory
cd /path/to/ClaudeBot

# Update Claude Bot
docker-compose down
git pull origin main
docker-compose build --no-cache
docker-compose up -d
docker-compose logs -f trading-bot
```

### **ClaudeTitan Bot (New)**
```bash
# Navigate to ClaudeTitan directory
cd ~/claudetitan

# Update ClaudeTitan Bot
docker-compose down
git pull origin master
docker-compose build --no-cache
docker-compose up -d
docker-compose logs -f trading-bot-enhanced
```

## 🚀 **Convenient Update Scripts**

I've created update scripts for both bots to make management easier:

### **Update ClaudeTitan Bot**
```bash
cd ~/claudetitan
chmod +x update_claudetitan.sh
./update_claudetitan.sh
```

### **Update Claude Bot**
```bash
cd /path/to/ClaudeBot
chmod +x update_claudebot.sh
./update_claudebot.sh
```

## 🔧 **Management Commands**

### **Check Status of Both Bots**
```bash
# Check Claude Bot
cd /path/to/ClaudeBot
docker-compose ps

# Check ClaudeTitan Bot
cd ~/claudetitan
docker-compose ps
```

### **View Logs**
```bash
# Claude Bot logs
cd /path/to/ClaudeBot
docker-compose logs -f trading-bot

# ClaudeTitan Bot logs
cd ~/claudetitan
docker-compose logs -f trading-bot-enhanced
```

### **Restart Bots**
```bash
# Restart Claude Bot
cd /path/to/ClaudeBot
docker-compose restart

# Restart ClaudeTitan Bot
cd ~/claudetitan
docker-compose restart
```

### **Stop Bots**
```bash
# Stop Claude Bot
cd /path/to/ClaudeBot
docker-compose down

# Stop ClaudeTitan Bot
cd ~/claudetitan
docker-compose down
```

### **Start Bots**
```bash
# Start Claude Bot
cd /path/to/ClaudeBot
docker-compose up -d

# Start ClaudeTitan Bot
cd ~/claudetitan
docker-compose up -d
```

## 🌐 **Access URLs**

### **Claude Bot**
- **Dashboard**: http://your-vps-ip:8000
- **Health Check**: http://your-vps-ip:8000/health
- **API**: http://your-vps-ip:8000/api/

### **ClaudeTitan Bot**
- **Dashboard**: http://your-vps-ip:8003
- **Health Check**: http://your-vps-ip:8003/health-enhanced
- **API**: http://your-vps-ip:8003/api/enhanced/

## 🔍 **Monitoring Both Bots**

### **Check All Running Containers**
```bash
# See all containers from both bots
docker ps

# Filter by bot name
docker ps --filter "name=trading-bot"
docker ps --filter "name=trading-bot-enhanced"
```

### **Resource Usage**
```bash
# Check resource usage
docker stats

# Check specific containers
docker stats trading-bot trading-bot-enhanced
```

### **System Resources**
```bash
# Check disk usage
df -h

# Check memory usage
free -h

# Check CPU usage
top
```

## 🆘 **Troubleshooting**

### **Port Conflicts**
```bash
# Check port usage
netstat -tulpn | grep :8000
netstat -tulpn | grep :8003
netstat -tulpn | grep :5432
netstat -tulpn | grep :5433
```

### **Container Issues**
```bash
# Check container logs
docker logs trading-bot
docker logs trading-bot-enhanced

# Restart specific container
docker restart trading-bot
docker restart trading-bot-enhanced
```

### **Database Issues**
```bash
# Check database containers
docker ps | grep postgres
docker ps | grep redis

# Check database logs
docker logs trading-postgres
docker logs trading-postgres-enhanced
```

## 📊 **Quick Status Check**

### **One-Line Status Check**
```bash
# Check both bots at once
echo "=== Claude Bot ===" && cd /path/to/ClaudeBot && docker-compose ps && echo "=== ClaudeTitan Bot ===" && cd ~/claudetitan && docker-compose ps
```

### **Health Check Both Bots**
```bash
# Claude Bot health
curl -s http://localhost:8000/health | jq .

# ClaudeTitan Bot health
curl -s http://localhost:8003/health-enhanced | jq .
```

## 🔄 **Update Workflow**

### **When You Need to Update Both Bots**
```bash
# 1. Update Claude Bot first
cd /path/to/ClaudeBot
./update_claudebot.sh

# 2. Update ClaudeTitan Bot
cd ~/claudetitan
./update_claudetitan.sh

# 3. Verify both are running
docker ps | grep trading-bot
```

### **When You Need to Update Only One Bot**
```bash
# Update only Claude Bot
cd /path/to/ClaudeBot
./update_claudebot.sh

# OR update only ClaudeTitan Bot
cd ~/claudetitan
./update_claudetitan.sh
```

## ⚠️ **Important Notes**

1. **Both bots can run simultaneously** - No conflicts
2. **Different ports** - Claude Bot (8000), ClaudeTitan Bot (8003)
3. **Different databases** - Completely isolated data
4. **Different networks** - No interference
5. **Independent updates** - Update one without affecting the other

## 🎯 **Best Practices**

1. **Always check status** before making changes
2. **Update one bot at a time** to avoid confusion
3. **Monitor logs** after updates
4. **Test health endpoints** after deployment
5. **Keep backups** of important configurations

**Both bots are now ready to run independently on your VPS!** 🚀
