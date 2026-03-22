from typing import Optional

from pydantic import BaseModel

from app.schemas import ModelInformationSchema, ValidationResultSchema


class UploadModelResponse(BaseModel):
    validation: ValidationResultSchema
    model_info: Optional[ModelInformationSchema] = None
