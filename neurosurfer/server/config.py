# server/config.py
"""
Server Configuration Module
============================

This module contains configuration settings for the Neurosurfer server,
including security, authentication, CORS, and database settings.

All settings can be overridden via environment variables for deployment flexibility.

Configuration Categories:
    - Security: SECRET_KEY for JWT signing
    - Authentication: Token expiration and cookie settings
    - CORS: Cross-origin resource sharing origins
    - Database: Database connection URL

Environment Variables:
    - SECRET_KEY: Secret key for JWT token signing (required in production)
    - ACCESS_TOKEN_EXPIRE_MINUTES: Token expiration time in minutes (default: 10080 = 7 days)
    - COOKIE_NAME: Name of the authentication cookie (default: "access_token")
    - COOKIE_SECURE: Whether to use secure cookies (HTTPS only) (default: false)
    - COOKIE_SAMESITE: SameSite cookie policy (default: "Lax")
    - CORS_ORIGINS: Comma-separated list of allowed origins (default: "http://localhost:5173")
    - DATABASE_URL: Database connection string (default: "sqlite:///./data/app.db")

Example:
    >>> # Set environment variables before import
    >>> import os
    >>> os.environ["SECRET_KEY"] = "my-secret-key-123"
    >>> os.environ["CORS_ORIGINS"] = "http://localhost:3000,https://myapp.com"
    >>> 
    >>> from neurosurfer.server.config import SECRET_KEY, CORS_ORIGINS
    >>> print(SECRET_KEY)
    'my-secret-key-123'
    >>> print(CORS_ORIGINS)
    ['http://localhost:3000', 'https://myapp.com']
"""
import os

SECRET_KEY = os.getenv("SECRET_KEY", "change-this-in-prod")

# 7 days access, sliding refresh will extend this
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "10080"))

# Cookies
COOKIE_NAME = os.getenv("COOKIE_NAME", "access_token")
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "false").lower() == "true"  # set TRUE in prod (HTTPS)
COOKIE_SAMESITE = os.getenv("COOKIE_SAMESITE", "Lax").capitalize()     # Lax/None/Strict

# CORS
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",") if o.strip()]

# Default Application Data Directory
from neurosurfer import CACHE_DIR
APP_DATA_PATH = os.path.join(CACHE_DIR, "app-storage")
os.makedirs(APP_DATA_PATH, exist_ok=True)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///" + APP_DATA_PATH + "/app.db")

