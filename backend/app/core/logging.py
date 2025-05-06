import logging
import sys
import json
from datetime import datetime
from typing import Any, Dict
from app.core.config import settings

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for logs"""
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "path": record.pathname
        }
        
        # Add extra fields if present
        if hasattr(record, "extra"):
            log_data.update(record.extra)
            
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": str(record.exc_info[0].__name__),
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info)
            }
            
        return json.dumps(log_data)

def setup_logging() -> None:
    """Configure application logging"""
    root_logger = logging.getLogger()
    
    # Remove existing handlers
    for handler in root_logger.handlers:
        root_logger.removeHandler(handler)
    
    # Set log level based on environment
    log_level = logging.DEBUG if settings.DEBUG else logging.INFO
    root_logger.setLevel(log_level)
    
    # Create console handler with JSON formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(console_handler)
    
    # Set levels for third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)  # Keep access logs
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)   # Keep error logs
    logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO if settings.DEBUG else logging.WARNING)  # SQL queries in debug
    logging.getLogger("alembic").setLevel(logging.INFO)  # Migration logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    # Create FastAPI logger
    fastapi_logger = logging.getLogger("fastapi")
    fastapi_logger.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)
    fastapi_logger.handlers = [console_handler]
    
    # Log startup message with configuration details
    root_logger.info(
        "Application logging configured",
        extra={
            "environment": settings.ENVIRONMENT,
            "debug": settings.DEBUG,
            "log_level": logging.getLevelName(log_level),
            "handlers": ["console"],
            "formatter": "JSON",
            "third_party_loggers": {
                "uvicorn.access": "INFO",
                "uvicorn.error": "INFO",
                "sqlalchemy.engine": "INFO" if settings.DEBUG else "WARNING",
                "alembic": "INFO",
                "httpx": "WARNING",
                "fastapi": "DEBUG" if settings.DEBUG else "INFO"
            }
        }
    ) 