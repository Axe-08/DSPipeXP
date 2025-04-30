from pydantic import BaseModel
from typing import Dict, Optional

class Song(BaseModel):
    id: str
    title: str
    artist: str
    duration: Optional[float] = None
    features: Optional[Dict] = None
    metadata: Optional[Dict] = None 