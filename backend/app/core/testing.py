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