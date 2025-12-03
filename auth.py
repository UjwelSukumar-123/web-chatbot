"""
Authentication Module
Handles user registration, login, and JWT token management
"""

import os
import json
import hashlib
import secrets
import bcrypt
from datetime import datetime, timedelta
from typing import Optional, Dict
from jose import JWTError, jwt
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

# Security scheme for FastAPI
security = HTTPBearer()

# User data file
USERS_FILE = "users.json"


def get_password_hash(password: str) -> str:
    """Hash a password using bcrypt"""
    # Encode password to bytes
    password_bytes = password.encode('utf-8')
    # Generate salt and hash password
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    # Return as string (decode bytes)
    return hashed.decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    try:
        # Encode both to bytes
        password_bytes = plain_password.encode('utf-8')
        hashed_bytes = hashed_password.encode('utf-8')
        # Verify password
        return bcrypt.checkpw(password_bytes, hashed_bytes)
    except Exception:
        return False


def load_users() -> Dict:
    """Load users from JSON file"""
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading users: {e}")
            return {}
    return {}


def save_users(users: Dict):
    """Save users to JSON file"""
    try:
        with open(USERS_FILE, 'w', encoding='utf-8') as f:
            json.dump(users, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving users: {e}")
        raise


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> Optional[Dict]:
    """Verify and decode a JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None


def register_user(email: str, password: str, organization_name: str = "", full_name: str = "") -> Dict:
    """
    Register a new user
    
    Args:
        email: User email (used as username)
        password: User password
        organization_name: Organization/company name
        full_name: User's full name
    
    Returns:
        Dict with user info (without password)
    
    Raises:
        HTTPException if user already exists
    """
    users = load_users()
    
    # Check if user already exists
    if email.lower() in users:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email already exists"
        )
    
    # Validate password strength
    if len(password) < 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 8 characters long"
        )
    
    # Create new user
    user_id = hashlib.sha256(email.lower().encode()).hexdigest()[:16]
    hashed_password = get_password_hash(password)
    
    user_data = {
        "user_id": user_id,
        "email": email.lower(),
        "password_hash": hashed_password,
        "organization_name": organization_name,
        "full_name": full_name,
        "created_at": datetime.utcnow().isoformat(),
        "is_active": True,
        "website_url": "",  # Each user can configure their own website
        "api_key": secrets.token_urlsafe(32)  # API key for plugin integration
    }
    
    users[email.lower()] = user_data
    save_users(users)
    
    # Return user info without password
    user_info = {k: v for k, v in user_data.items() if k != "password_hash"}
    return user_info


def authenticate_user(email: str, password: str) -> Optional[Dict]:
    """
    Authenticate a user with email and password
    
    Returns:
        User dict if authentication successful, None otherwise
    """
    users = load_users()
    email_lower = email.lower()
    
    if email_lower not in users:
        return None
    
    user = users[email_lower]
    
    # Check if user is active
    if not user.get("is_active", True):
        return None
    
    # Verify password
    if not verify_password(password, user["password_hash"]):
        return None
    
    # Return user info without password
    user_info = {k: v for k, v in user.items() if k != "password_hash"}
    return user_info


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict:
    """
    Get current authenticated user from JWT token
    
    This is a dependency that can be used in FastAPI routes
    """
    token = credentials.credentials
    payload = verify_token(token)
    
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    email: str = payload.get("sub")
    if email is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    users = load_users()
    user = users.get(email.lower())
    
    if user is None or not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Return user info without password
    user_info = {k: v for k, v in user.items() if k != "password_hash"}
    return user_info


def get_user_by_email(email: str) -> Optional[Dict]:
    """Get user by email"""
    users = load_users()
    user = users.get(email.lower())
    if user:
        return {k: v for k, v in user.items() if k != "password_hash"}
    return None


def update_user_website(email: str, website_url: str):
    """Update user's website URL"""
    users = load_users()
    email_lower = email.lower()
    
    if email_lower not in users:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    users[email_lower]["website_url"] = website_url
    save_users(users)
    
    return {k: v for k, v in users[email_lower].items() if k != "password_hash"}

