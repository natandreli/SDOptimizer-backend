from typing import Any, Dict, List

from pydantic import BaseModel, Field


class ValidationResultSchema(BaseModel):
    """Schema for successful validation results."""

    is_valid: bool
    format: str = "mdl"
    checks_passed: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
