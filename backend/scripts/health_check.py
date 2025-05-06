import asyncio
import logging
import sys
from app.core.testing import verify_database_health

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

async def run_health_check():
    try:
        logging.info("Starting database health check...")
        healthy = await verify_database_health()
        if not healthy:
            logging.error("Database health check failed")
            sys.exit(1)
        logging.info("Database health check passed")
    except Exception as e:
        logging.error(f"Unhandled exception in health check: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(run_health_check()) 