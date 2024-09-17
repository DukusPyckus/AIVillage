from typing import List, Optional
from pydantic import BaseModel

class ModelReference(BaseModel):
    name: str
    path: Optional[str] = None

class MergeKitConfig(BaseModel):
    merge_method: str
    models: List[ModelReference]
    parameters: Optional[dict] = None