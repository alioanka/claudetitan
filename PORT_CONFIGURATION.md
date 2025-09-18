# Port Configuration - Avoiding Conflicts with Claude Bot

## Current Port Allocation

### Claude Bot (Existing)
- **Port 8000**: Main dashboard and API
- **Port 9090**: Health check server
- **Port 5432**: PostgreSQL database (`trading_bot`)
- **Port 6379**: Redis cache (database 0)

### Trading Bot (New)
- **Port 8002**: Trading bot engine (internal port 8000)
- **Port 8003**: Trading bot dashboard (internal port 8000)
- **Port 8080**: Nginx HTTP (internal port 80)
- **Port 8443**: Nginx HTTPS (internal port 443)
- **Port 9091**: Prometheus monitoring (internal port 9090)
- **Port 3001**: Grafana dashboard (internal port 3000)
- **Port 5433**: PostgreSQL database (`trading_bot_enhanced`)
- **Port 6380**: Redis cache (database 1)

## Service Access URLs

### Claude Bot
- Dashboard: `http://localhost:8000`
- Health Check: `http://localhost:9090/health`

### Trading Bot
- Dashboard: `http://localhost:8003` or `https://localhost:8443`
- API: `http://localhost:8003/api`
- Engine: `http://localhost:8002`
- Prometheus: `http://localhost:9091`
- Grafana: `http://localhost:3001`

## Docker Compose Integration

Both bots can run simultaneously using different Docker Compose files:

### Claude Bot
```bash
cd C:\Users\HP\Desktop\ClaudeBot
docker-compose up -d
```

### Trading Bot
```bash
cd C:\Users\HP\Desktop\ClaudeBot
docker-compose up -d
```

## Network Isolation

Each bot uses its own Docker network:
- **Claude Bot**: `trading_network` (existing)
- **Trading Bot**: `trading-network` (new)

This ensures complete isolation between the two systems.

## Database Separation

Both bots use completely separate databases:

### Claude Bot Database
- **PostgreSQL**: Port 5432, Database: `trading_bot`, User: `trader`
- **Redis**: Port 6379, Database: 0

### Trading Bot Database
- **PostgreSQL**: Port 5433, Database: `trading_bot_enhanced`, User: `trading_enhanced`
- **Redis**: Port 6380, Database: 1

**No shared resources** - complete isolation between the two systems.

## Deployment Order

1. **Start Claude Bot first** (uses standard ports)
2. **Start Trading Bot second** (uses alternative ports)

## Health Checks

### Claude Bot Health
```bash
curl http://localhost:8000/health
curl http://localhost:9090/health
```

### Trading Bot Health
```bash
curl http://localhost:8003/health
curl http://localhost:8002/health
```

## Monitoring

Both bots can be monitored independently:
- Claude Bot: `http://localhost:9090` (Prometheus)
- Trading Bot: `http://localhost:9091` (Prometheus)

## Troubleshooting

If you encounter port conflicts:

1. **Check running services**:
   ```bash
   netstat -tulpn | grep :8000
   netstat -tulpn | grep :9090
   ```

2. **Stop conflicting services**:
   ```bash
   docker-compose down
   ```

3. **Restart in correct order**:
   ```bash
   # Claude Bot first
   cd C:\Users\HP\Desktop\ClaudeBot
   docker-compose up -d
   
   # Trading Bot second
   cd C:\Users\HP\Desktop\ClaudeTitan
   docker-compose up -d
   ```

## Security Considerations

- Both bots use separate authentication systems
- No cross-communication between bot networks
- Independent SSL certificates
- Separate logging and monitoring

This configuration ensures both trading bots can run simultaneously without conflicts while maintaining complete isolation and security.
