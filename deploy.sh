#!/bin/bash

# Trading Bot Deployment Script
# This script deploys the trading bot to a VPS

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="trading-bot-enhanced"
DOCKER_COMPOSE_FILE="docker-compose.yml"
ENV_FILE=".env"

# Functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        error "This script should not be run as root"
        exit 1
    fi
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if .env file exists
    if [[ ! -f "$ENV_FILE" ]]; then
        warning ".env file not found. Creating from template..."
        cp env_example.txt .env
        warning "Please edit .env file with your configuration before running again."
        exit 1
    fi
    
    success "Prerequisites check passed"
}

# Create necessary directories
create_directories() {
    log "Creating necessary directories..."
    
    mkdir -p data logs models ssl static
    chmod 755 data logs models ssl static
    
    success "Directories created"
}

# Generate SSL certificates (self-signed for development)
generate_ssl_certificates() {
    log "Generating SSL certificates..."
    
    if [[ ! -f "ssl/cert.pem" ]] || [[ ! -f "ssl/key.pem" ]]; then
        openssl req -x509 -newkey rsa:4096 -keyout ssl/key.pem -out ssl/cert.pem -days 365 -nodes -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
        chmod 600 ssl/key.pem
        chmod 644 ssl/cert.pem
        success "SSL certificates generated"
    else
        warning "SSL certificates already exist, skipping generation"
    fi
}

# Create Prometheus configuration
create_prometheus_config() {
    log "Creating Prometheus configuration..."
    
    cat > prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'trading-bot'
    static_configs:
      - targets: ['trading-bot:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'dashboard'
    static_configs:
      - targets: ['dashboard:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s
EOF

    success "Prometheus configuration created"
}

# Build and start services
deploy_services() {
    log "Building and starting services..."
    
    # Stop existing services
    docker-compose down --remove-orphans
    
    # Build and start services
    docker-compose up -d --build
    
    success "Services deployed"
}

# Wait for services to be ready
wait_for_services() {
    log "Waiting for services to be ready..."
    
    # Wait for dashboard
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -f http://localhost:8003/health &> /dev/null; then
            success "Dashboard is ready"
            break
        fi
        
        if [[ $attempt -eq $max_attempts ]]; then
            error "Dashboard failed to start after $max_attempts attempts"
            exit 1
        fi
        
        log "Waiting for dashboard... (attempt $attempt/$max_attempts)"
        sleep 10
        ((attempt++))
    done
}

# Show deployment status
show_status() {
    log "Deployment Status:"
    echo ""
    
    # Docker services status
    docker-compose ps
    
    echo ""
    log "Service URLs:"
    echo "  Trading Bot Dashboard: https://localhost:8003"
    echo "  Trading Bot API: https://localhost:8003/api"
    echo "  Trading Bot Engine: http://localhost:8002"
    echo "  Prometheus: http://localhost:9091"
    echo "  Grafana: http://localhost:3001 (admin/admin)"
    echo "  Nginx (HTTP): http://localhost:8080"
    echo "  Nginx (HTTPS): https://localhost:8443"
    echo ""
    
    log "Useful commands:"
    echo "  View logs: docker-compose logs -f"
    echo "  Stop services: docker-compose down"
    echo "  Restart services: docker-compose restart"
    echo "  Update services: docker-compose pull && docker-compose up -d"
    echo ""
}

# Main deployment function
main() {
    log "Starting Trading Bot deployment..."
    
    check_root
    check_prerequisites
    create_directories
    generate_ssl_certificates
    create_prometheus_config
    deploy_services
    wait_for_services
    show_status
    
    success "Trading Bot deployed successfully!"
    log "You can now access the dashboard at https://localhost:8003"
}

# Run main function
main "$@"
