#!/bin/bash

# Claude Bot Update Script
# This script updates the existing Claude Bot

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Main update function
update_claudebot() {
    log "Starting Claude Bot update..."
    
    # Check if we're in the right directory
    if [ ! -f "docker-compose.yml" ]; then
        error "docker-compose.yml not found. Please run this script from the Claude Bot directory."
        exit 1
    fi
    
    # Stop containers
    log "Stopping Claude Bot containers..."
    docker-compose down
    
    # Pull latest changes
    log "Pulling latest changes from GitHub..."
    git pull origin main
    
    # Build with no cache
    log "Building containers with no cache..."
    docker-compose build --no-cache
    
    # Start containers
    log "Starting Claude Bot containers..."
    docker-compose up -d
    
    # Wait a moment for containers to start
    sleep 5
    
    # Check status
    log "Checking container status..."
    docker-compose ps
    
    # Show logs
    log "Showing recent logs..."
    docker-compose logs --tail=20
    
    success "Claude Bot updated successfully!"
    log "Dashboard: http://localhost:8000"
    log "Health check: http://localhost:8000/health"
    log "To view live logs: docker-compose logs -f trading-bot"
}

# Run update
update_claudebot
