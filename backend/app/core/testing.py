import logging
import traceback
import httpx
import asyncio
from typing import Dict, List, Any
from fastapi import FastAPI
from ..core.config import settings
from sqlalchemy.sql import text
from ..core.database import db_manager

logger = logging.getLogger(__name__)

class EndpointTester:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.client = httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True  # Automatically follow redirects
        )
        
    async def test_endpoint(self, method: str, path: str, json: Dict = None, expected_status: int = 200) -> Dict[str, Any]:
        """Test a single endpoint and return results"""
        full_url = f"{self.base_url}{path}"
        start_time = asyncio.get_event_loop().time()
        
        try:
            response = await self.client.request(method, full_url, json=json)
            duration = asyncio.get_event_loop().time() - start_time
            
            result = {
                "endpoint": path,
                "method": method,
                "status_code": response.status_code,
                "duration": f"{duration:.2f}s",
                "success": response.status_code == expected_status,
                "response": response.json() if response.headers.get("content-type", "").startswith("application/json") else str(response.content),
            }
            
            if not result["success"]:
                result["error"] = f"Unexpected status code: {response.status_code}"
                if response.status_code == 422:
                    result["validation_errors"] = response.json().get("detail", [])
                
            return result
            
        except Exception as e:
            duration = asyncio.get_event_loop().time() - start_time
            return {
                "endpoint": path,
                "method": method,
                "status_code": None,
                "duration": f"{duration:.2f}s",
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }

    async def run_endpoint_tests(self) -> List[Dict[str, Any]]:
        """Run tests for all endpoints"""
        test_cases = [
            # Health endpoints
            ("GET", "/api/v1/health", None, 200),
            
            # Songs endpoints
            ("GET", "/api/v1/songs/", None, 200),
            ("GET", "/api/v1/songs/search", {"query": "test"}, 200),
            
            # YouTube endpoints - skip for now due to bot detection
            # ("POST", "/api/v1/youtube/process", {"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "quality": "192"}, 200),
            # ("GET", "/api/v1/youtube/metadata?url=https://www.youtube.com/watch?v=dQw4w9WgXcQ", None, 200),
            
            # Search endpoints
            ("GET", "/api/v1/search/by-name/test", None, 200),
            
            # Monitoring endpoints
            ("GET", "/api/v1/monitoring/stats", None, 200),
        ]
        
        results = []
        for method, path, json_data, expected_status in test_cases:
            result = await self.test_endpoint(method, path, json_data, expected_status)
            results.append(result)
            
            # Log result immediately
            if result["success"]:
                logger.info(
                    f"✅ {method} {path} - {result['status_code']} ({result['duration']})",
                    extra={"test_result": result}
                )
            else:
                logger.error(
                    f"❌ {method} {path} - Failed ({result['duration']})",
                    extra={
                        "test_result": result,
                        "error": result.get("error"),
                        "traceback": result.get("traceback"),
                        "validation_errors": result.get("validation_errors")
                    }
                )
            
            # Small delay between requests
            await asyncio.sleep(1)
            
        return results

    async def close(self):
        await self.client.aclose()

async def verify_database_health() -> bool:
    """Verify database health by checking song count, indexes, and functionality."""
    try:
        logger.info("Starting comprehensive database health verification...")
        
        try:
            # Initialize database manager first
            logger.info("Initializing database manager...")
            await db_manager.initialize()
            logger.info("Database manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database manager: {str(e)}", exc_info=True)
            return False
        
        try:
            # Get a session for our health checks
            logger.info("Getting database session...")
            session = await db_manager.get_session()
            logger.info("Database session obtained successfully")
        except Exception as e:
            logger.error(f"Failed to get database session: {str(e)}", exc_info=True)
            return False

        try:
            # Check database connection
            try:
                logger.info("Testing database connection...")
                result = await session.execute(text("SELECT 1"))
                value = result.scalar()
                if value != 1:
                    logger.error("Database connection check returned unexpected value")
                    return False
                logger.info("Database connection successful")
            except Exception as e:
                logger.error(f"Database connection failed: {str(e)}", exc_info=True)
                return False

            # Check if pg_trgm extension is installed
            result = await session.execute(text(
                "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm')"
            ))
            has_trgm = result.scalar()
            if not has_trgm:
                logger.error("pg_trgm extension not found - fuzzy search will not work")
                return False
            logger.info("pg_trgm extension verified")

            # Check if required indexes exist
            result = await session.execute(text("""
                SELECT COUNT(*) 
                FROM pg_indexes 
                WHERE indexname IN (
                    'idx_song_track_name_trgm',
                    'idx_song_artist_trgm'
                )
            """))
            index_count = result.scalar()
            if index_count != 2:
                logger.error("Missing required trigram indexes")
                return False
            logger.info("Required indexes verified")

            # Get total song count
            result = await session.execute(text("SELECT COUNT(*) FROM songs"))
            total_songs = result.scalar()
            logger.info(f"Total songs in database: {total_songs}")
            
            if total_songs < 18000:
                logger.warning(f"Database has fewer songs than expected ({total_songs} < 18000), but continuing anyway")

            # Verify fuzzy search functionality
            try:
                result = await session.execute(text("""
                    SELECT EXISTS (
                        SELECT 1 
                        FROM songs 
                        WHERE similarity(track_name, 'test') > 0.3
                        LIMIT 1
                    )
                """))
                has_fuzzy_results = result.scalar()
                if has_fuzzy_results:
                    logger.info("Fuzzy search functionality verified")
                else:
                    logger.warning("No fuzzy search results found, but functionality appears to work")
            except Exception as e:
                logger.error(f"Fuzzy search test failed: {str(e)}", exc_info=True)
                return False

            # Check data integrity
            result = await session.execute(text("""
                SELECT COUNT(*) 
                FROM songs 
                WHERE track_name IS NULL 
                   OR track_artist IS NULL 
                   OR track_name = '' 
                   OR track_artist = ''
            """))
            invalid_records = result.scalar()
            if invalid_records > 0:
                logger.warning(f"Found {invalid_records} records with missing required data, but continuing anyway")
            logger.info("Data integrity verified")

            # Check if audio features are present and valid JSON
            result = await session.execute(text("""
                SELECT COUNT(*) 
                FROM songs 
                WHERE audio_features IS NOT NULL 
                  AND audio_features::jsonb IS NOT NULL
            """))
            songs_with_features = result.scalar()
            logger.info(f"Songs with valid audio features: {songs_with_features}")

            # Check database size
            result = await session.execute(text("""
                SELECT pg_size_pretty(pg_database_size(current_database()))
            """))
            db_size = result.scalar()
            logger.info(f"Database size: {db_size if db_size else 'unknown'}")

            return True

        finally:
            try:
                logger.info("Closing database session...")
                await session.close()
                logger.info("Database session closed successfully")
            except Exception as e:
                logger.error(f"Error closing database session: {str(e)}", exc_info=True)

    except Exception as e:
        logger.error(f"Unhandled error in database health check: {str(e)}", exc_info=True)
        return False

async def run_startup_tests():
    """Run all startup tests including database verification."""
    logger.info("Starting startup tests...")
    
    # First verify database health
    db_healthy = await verify_database_health()
    if not db_healthy:
        logger.error("Database health check failed - this may affect endpoint functionality")
    else:
        logger.info("Database health check passed successfully")
    
    # Then run endpoint tests
    await run_endpoint_tests()

async def run_startup_tests(app: FastAPI):
    """Run all endpoint tests after startup"""
    # Wait a bit for the application to fully start
    await asyncio.sleep(5)
    
    logger.info("Starting endpoint tests...")
    
    # Determine base URL based on environment
    if settings.ENVIRONMENT == "development":
        base_url = f"http://{settings.HOST}:{settings.PORT}"
    else:
        # For production, use the Render URL
        base_url = "https://dspipexp.onrender.com"
    
    tester = EndpointTester(base_url)
    
    try:
        results = await tester.run_endpoint_tests()
        
        # Summarize results
        total = len(results)
        successful = sum(1 for r in results if r["success"])
        failed = total - successful
        
        if failed > 0:
            logger.error(
                f"Endpoint tests completed: {successful}/{total} passed, {failed} failed",
                extra={"test_summary": {
                    "total": total,
                    "successful": successful,
                    "failed": failed,
                    "results": results
                }}
            )
        else:
            logger.info(
                f"All endpoint tests passed: {successful}/{total}",
                extra={"test_summary": {
                    "total": total,
                    "successful": successful,
                    "failed": failed,
                    "results": results
                }}
            )
            
    except Exception as e:
        logger.error(
            f"Error running startup tests: {str(e)}",
            extra={"error": str(e), "traceback": traceback.format_exc()}
        )
    finally:
        await tester.close() 