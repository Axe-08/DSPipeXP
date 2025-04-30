import asyncio
import logging
from datetime import datetime
from typing import Optional
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.utils.cleanup import storage_manager
import httpx

logger = logging.getLogger(__name__)

class BackgroundTasks:
    def __init__(self, app: FastAPI):
        self.app = app
        self.cleanup_task: Optional[asyncio.Task] = None
        self.keepalive_task: Optional[asyncio.Task] = None
        self.is_running = False

    async def cleanup_worker(self):
        """Periodic cleanup worker"""
        while self.is_running:
            try:
                async with AsyncSession(get_db()) as db:
                    await storage_manager.check_and_cleanup(db)
            except Exception as e:
                logger.error(f"Error in cleanup worker: {str(e)}")
            
            # Run cleanup every hour
            await asyncio.sleep(3600)

    async def keepalive_worker(self):
        """Keep the service alive by pinging health endpoint"""
        while self.is_running:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        "http://localhost:8000/api/v1/health"
                    )
                    logger.debug(f"Keepalive ping: {response.status_code}")
            except Exception as e:
                logger.error(f"Error in keepalive worker: {str(e)}")
            
            # Ping every 10 minutes
            await asyncio.sleep(600)

    async def start(self):
        """Start background tasks"""
        self.is_running = True
        
        # Start cleanup task
        self.cleanup_task = asyncio.create_task(
            self.cleanup_worker(),
            name="cleanup_worker"
        )
        
        # Start keepalive task
        self.keepalive_task = asyncio.create_task(
            self.keepalive_worker(),
            name="keepalive_worker"
        )
        
        logger.info("Background tasks started")

    async def stop(self):
        """Stop background tasks"""
        self.is_running = False
        
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
            
        if self.keepalive_task:
            self.keepalive_task.cancel()
            try:
                await self.keepalive_task
            except asyncio.CancelledError:
                pass
            
        logger.info("Background tasks stopped") 