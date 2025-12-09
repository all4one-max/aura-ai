"""
Configuration management for the application.
Loads all environment variables from .env file.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Database - support both SQLite (dev) and PostgreSQL (prod)
DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///database.db")

# API Keys
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
GOOGLE_VERTEX_AI_PROJECT_ID: str = os.getenv("GOOGLE_VERTEX_AI_PROJECT_ID", "")
ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY", "")
SERPAPI_API_KEY: str = os.getenv("SERPAPI_API_KEY", "")
GOOGLE_APPLICATION_CREDENTIALS: str = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
