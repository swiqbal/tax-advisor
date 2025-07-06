"""
Settings configuration for AI Tax Advisor Demo
Environment-based configuration management for standalone AI demo
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Configure debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Get the absolute path to the project root
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
logger.debug(f"Project root directory: {PROJECT_ROOT}")

# Load environment variables from .env file
env_path = PROJECT_ROOT / '.env'
logger.debug(f"Looking for .env file at: {env_path}")

if env_path.exists():
    logger.debug(".env file found, loading variables...")
    load_dotenv(env_path)
    logger.debug("Environment variables loaded from .env")
else:
    logger.warning(".env file not found. Using environment_sample.txt as fallback.")
    sample_path = PROJECT_ROOT / 'environment_sample.txt'
    if sample_path.exists():
        load_dotenv(sample_path)
        logger.debug("Environment variables loaded from environment_sample.txt")
    else:
        logger.error("No environment file found!")

class Settings:
    """Application settings for AI Tax Advisor Demo"""
    
    def __init__(self):
        """Initialize settings and create necessary directories"""
        logger.debug("Initializing Settings...")
        
        # Create logs directory if it doesn't exist
        Path("logs").mkdir(exist_ok=True)
        
        # Print current configuration for debugging
        logger.debug("\nCurrent Configuration:")
        logger.debug(f"API Key Set: {'Yes' if self.GEMINI_API_KEY else 'No'}")
        logger.debug(f"API Key Length: {len(self.GEMINI_API_KEY) if self.GEMINI_API_KEY else 0}")
        logger.debug(f"Model: {self.GEMINI_MODEL}")
        logger.debug(f"Max Tokens: {self.GEMINI_MAX_TOKENS}")
        logger.debug(f"Temperature: {self.GEMINI_TEMPERATURE}")
        
        # Print all environment variables for debugging (excluding sensitive data)
        logger.debug("\nAll environment variables:")
        for key, value in os.environ.items():
            if 'KEY' in key or 'SECRET' in key or 'PASSWORD' in key:
                logger.debug(f"{key}: [HIDDEN]")
            else:
                logger.debug(f"{key}: {value}")
    
    # Application settings
    APP_NAME: str = "AI Tax Advisor Demo"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # Security settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", "ai-demo-secret-key")
    
    # AI/LLM settings - Gemini Flash 2.0 Pro
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
    GEMINI_MAX_TOKENS: int = int(os.getenv("GEMINI_MAX_TOKENS", "8192"))
    GEMINI_TEMPERATURE: float = float(os.getenv("GEMINI_TEMPERATURE", "0.7"))
    
    # API Rate Limiting settings
    GEMINI_REQUESTS_PER_MINUTE: int = int(os.getenv("GEMINI_REQUESTS_PER_MINUTE", "60"))
    GEMINI_REQUESTS_PER_HOUR: int = int(os.getenv("GEMINI_REQUESTS_PER_HOUR", "1000"))
    GEMINI_MAX_RETRIES: int = int(os.getenv("GEMINI_MAX_RETRIES", "3"))
    GEMINI_RETRY_DELAY: int = int(os.getenv("GEMINI_RETRY_DELAY", "1"))
    
    # AI Safety settings
    GEMINI_SAFETY_THRESHOLD: str = os.getenv("GEMINI_SAFETY_THRESHOLD", "BLOCK_MEDIUM_AND_ABOVE")
    GEMINI_RESPONSE_VALIDATION: bool = os.getenv("GEMINI_RESPONSE_VALIDATION", "true").lower() == "true"
    
    # Logging settings
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "ai_demo.log")
    
    def validate_config(self) -> bool:
        """Validate configuration settings"""
        logger.debug("Validating configuration...")
        
        # Validate Gemini API configuration
        if not self.GEMINI_API_KEY:
            logger.error("GEMINI_API_KEY not set. AI features will be disabled.")
            return False
        
        # Validate rate limiting settings
        if self.GEMINI_REQUESTS_PER_MINUTE <= 0 or self.GEMINI_REQUESTS_PER_HOUR <= 0:
            logger.error("Invalid rate limiting configuration")
            raise ValueError("Invalid rate limiting configuration")
        
        logger.debug("Configuration validation successful")
        return True

# Global settings instance
settings = Settings()

# Validate configuration on import
try:
    settings.validate_config()
except Exception as e:
    logger.error(f"Configuration validation error: {e}")
    raise 