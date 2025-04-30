import logging
import traceback
import httpx
import asyncio
from typing import Dict, List, Any
from fastapi import FastAPI
from ..core.config import settings

logger = logging.getLogger(__name__)

class EndpointTester:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.client = httpx.AsyncClient(timeout=30.0)
        
    async def test_endpoint(self, method: str, path: str, json: Dict = None) -> Dict[str, Any]:
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
                "success": 200 <= response.status_code < 300,
                "response": response.json() if response.headers.get("content-type", "").startswith("application/json") else str(response.content),
            }
            
            if not result["success"]:
                result["error"] = f"Unexpected status code: {response.status_code}"
                
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
            ("GET", "/api/v1/health"),
            
            # Songs endpoints
            ("GET", "/api/v1/songs/"),
            ("GET", "/api/v1/songs/search?query=test"),
            ("POST", "/api/v1/songs/youtube", {"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}),
            
            # YouTube endpoints
            ("POST", "/api/v1/youtube/process", {"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "quality": "192"}),
            ("GET", "/api/v1/youtube/metadata?url=https://www.youtube.com/watch?v=dQw4w9WgXcQ"),
            
            # Search endpoints
            ("GET", "/api/v1/search/by-name/test"),
            
            # Monitoring endpoints
            ("GET", "/api/v1/monitoring/stats"),
        ]
        
        results = []
        for method, path, *args in test_cases:
            json_data = args[0] if args else None
            result = await self.test_endpoint(method, path, json_data)
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
                        "traceback": result.get("traceback")
                    }
                )
            
            # Small delay between requests
            await asyncio.sleep(1)
            
        return results

    async def close(self):
        await self.client.aclose()

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