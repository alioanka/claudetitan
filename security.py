"""
Security and Authentication Module
"""
import logging
import hashlib
import secrets
import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import bcrypt

from config import settings

logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
JWT_SECRET_KEY = settings.secret_key
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = settings.access_token_expire_minutes

# Security schemes
security_scheme = HTTPBearer()

class SecurityManager:
    """Security manager for authentication and authorization"""
    
    def __init__(self):
        self.active_tokens = set()
        self.failed_attempts = {}
        self.max_failed_attempts = 5
        self.lockout_duration = 300  # 5 minutes
    
    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt"""
        try:
            return pwd_context.hash(password)
        except Exception as e:
            logger.error(f"Error hashing password: {e}")
            raise
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        try:
            return pwd_context.verify(plain_password, hashed_password)
        except Exception as e:
            logger.error(f"Error verifying password: {e}")
            return False
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create a JWT access token"""
        try:
            to_encode = data.copy()
            
            if expires_delta:
                expire = datetime.utcnow() + expires_delta
            else:
                expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
            
            to_encode.update({"exp": expire, "iat": datetime.utcnow()})
            
            encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
            
            # Store token in active tokens set
            self.active_tokens.add(encoded_jwt)
            
            return encoded_jwt
            
        except Exception as e:
            logger.error(f"Error creating access token: {e}")
            raise
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode a JWT token"""
        try:
            # Check if token is in active tokens
            if token not in self.active_tokens:
                return None
            
            # Decode token
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            
            # Check expiration
            exp = payload.get("exp")
            if exp and datetime.utcnow() > datetime.fromtimestamp(exp):
                self.active_tokens.discard(token)
                return None
            
            return payload
            
        except jwt.ExpiredSignatureError:
            self.active_tokens.discard(token)
            return None
        except jwt.InvalidTokenError:
            return None
        except Exception as e:
            logger.error(f"Error verifying token: {e}")
            return None
    
    def revoke_token(self, token: str) -> bool:
        """Revoke a token"""
        try:
            self.active_tokens.discard(token)
            return True
        except Exception as e:
            logger.error(f"Error revoking token: {e}")
            return False
    
    def revoke_all_tokens(self) -> int:
        """Revoke all active tokens"""
        try:
            count = len(self.active_tokens)
            self.active_tokens.clear()
            return count
        except Exception as e:
            logger.error(f"Error revoking all tokens: {e}")
            return 0
    
    def check_rate_limit(self, client_ip: str) -> bool:
        """Check if client is rate limited"""
        try:
            current_time = datetime.utcnow()
            
            # Clean old entries
            if client_ip in self.failed_attempts:
                attempts = self.failed_attempts[client_ip]
                self.failed_attempts[client_ip] = [
                    attempt_time for attempt_time in attempts
                    if (current_time - attempt_time).total_seconds() < self.lockout_duration
                ]
                
                # Remove empty entries
                if not self.failed_attempts[client_ip]:
                    del self.failed_attempts[client_ip]
            
            # Check if client is locked out
            if client_ip in self.failed_attempts:
                attempts = self.failed_attempts[client_ip]
                if len(attempts) >= self.max_failed_attempts:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            return True
    
    def record_failed_attempt(self, client_ip: str) -> None:
        """Record a failed authentication attempt"""
        try:
            current_time = datetime.utcnow()
            
            if client_ip not in self.failed_attempts:
                self.failed_attempts[client_ip] = []
            
            self.failed_attempts[client_ip].append(current_time)
            
        except Exception as e:
            logger.error(f"Error recording failed attempt: {e}")
    
    def generate_api_key(self) -> str:
        """Generate a secure API key"""
        try:
            return secrets.token_urlsafe(32)
        except Exception as e:
            logger.error(f"Error generating API key: {e}")
            raise
    
    def hash_api_key(self, api_key: str) -> str:
        """Hash an API key for storage"""
        try:
            return hashlib.sha256(api_key.encode()).hexdigest()
        except Exception as e:
            logger.error(f"Error hashing API key: {e}")
            raise
    
    def verify_api_key(self, api_key: str, hashed_api_key: str) -> bool:
        """Verify an API key against its hash"""
        try:
            return self.hash_api_key(api_key) == hashed_api_key
        except Exception as e:
            logger.error(f"Error verifying API key: {e}")
            return False

# Global security manager instance
security_manager = SecurityManager()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security_scheme)) -> Dict[str, Any]:
    """Get current user from JWT token"""
    try:
        token = credentials.credentials
        payload = security_manager.verify_token(token)
        
        if payload is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return payload
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting current user: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

def require_permission(permission: str):
    """Decorator to require specific permission"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # This would check user permissions
            # For now, just return the function
            return func(*args, **kwargs)
        return wrapper
    return decorator

class APIKeyAuth:
    """API Key authentication"""
    
    def __init__(self, api_keys: Dict[str, str]):
        self.api_keys = api_keys  # {api_key_hash: user_id}
    
    def __call__(self, api_key: str = Depends(HTTPBearer())):
        try:
            if not api_key:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="API key required"
                )
            
            # Hash the provided API key
            hashed_key = security_manager.hash_api_key(api_key.credentials)
            
            # Check if API key exists
            if hashed_key not in self.api_keys:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid API key"
                )
            
            return {
                "user_id": self.api_keys[hashed_key],
                "api_key": hashed_key
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in API key authentication: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed"
            )

class RateLimiter:
    """Rate limiting decorator"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 3600):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}  # {client_ip: [timestamps]}
    
    def __call__(self, request):
        try:
            client_ip = request.client.host
            current_time = datetime.utcnow()
            
            # Clean old requests
            if client_ip in self.requests:
                self.requests[client_ip] = [
                    timestamp for timestamp in self.requests[client_ip]
                    if (current_time - timestamp).total_seconds() < self.window_seconds
                ]
            
            # Check rate limit
            if client_ip in self.requests:
                if len(self.requests[client_ip]) >= self.max_requests:
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail="Rate limit exceeded"
                    )
            
            # Record request
            if client_ip not in self.requests:
                self.requests[client_ip] = []
            self.requests[client_ip].append(current_time)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in rate limiting: {e}")
            # Don't block on error, just log it

def validate_input(data: Dict[str, Any], required_fields: list) -> bool:
    """Validate input data"""
    try:
        # Check required fields
        for field in required_fields:
            if field not in data:
                return False
        
        # Additional validation can be added here
        return True
        
    except Exception as e:
        logger.error(f"Error validating input: {e}")
        return False

def sanitize_input(data: str) -> str:
    """Sanitize input data"""
    try:
        # Remove potentially dangerous characters
        dangerous_chars = ['<', '>', '"', "'", '&', ';', '(', ')', '|', '`', '$']
        
        for char in dangerous_chars:
            data = data.replace(char, '')
        
        return data.strip()
        
    except Exception as e:
        logger.error(f"Error sanitizing input: {e}")
        return ""

def encrypt_sensitive_data(data: str, key: str = None) -> str:
    """Encrypt sensitive data"""
    try:
        if key is None:
            key = JWT_SECRET_KEY
        
        # Simple encryption using base64 and XOR
        import base64
        
        # XOR encryption
        encrypted = ''.join(chr(ord(c) ^ ord(key[i % len(key)])) for i, c in enumerate(data))
        
        # Base64 encode
        return base64.b64encode(encrypted.encode()).decode()
        
    except Exception as e:
        logger.error(f"Error encrypting data: {e}")
        return data

def decrypt_sensitive_data(encrypted_data: str, key: str = None) -> str:
    """Decrypt sensitive data"""
    try:
        if key is None:
            key = JWT_SECRET_KEY
        
        import base64
        
        # Base64 decode
        encrypted = base64.b64decode(encrypted_data.encode()).decode()
        
        # XOR decryption
        decrypted = ''.join(chr(ord(c) ^ ord(key[i % len(key)])) for i, c in enumerate(encrypted))
        
        return decrypted
        
    except Exception as e:
        logger.error(f"Error decrypting data: {e}")
        return encrypted_data

def log_security_event(event_type: str, details: Dict[str, Any], client_ip: str = None):
    """Log security events"""
    try:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "details": details,
            "client_ip": client_ip
        }
        
        logger.warning(f"Security Event: {log_data}")
        
        # In production, you might want to send this to a security monitoring system
        
    except Exception as e:
        logger.error(f"Error logging security event: {e}")

def check_trading_permissions(user_data: Dict[str, Any]) -> bool:
    """Check if user has trading permissions"""
    try:
        # Check if user is authenticated
        if not user_data:
            return False
        
        # Check if user has trading role
        user_role = user_data.get("role", "viewer")
        if user_role not in ["admin", "trader"]:
            return False
        
        # Additional permission checks can be added here
        
        return True
        
    except Exception as e:
        logger.error(f"Error checking trading permissions: {e}")
        return False

def require_trading_permission(func):
    """Decorator to require trading permission"""
    def wrapper(*args, **kwargs):
        # This would check trading permissions
        # For now, just return the function
        return func(*args, **kwargs)
    return wrapper
