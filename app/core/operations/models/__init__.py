from __future__ import annotations

from fastapi import UploadFile

from app.api.routers.models.response_schemas import UploadModelResponse
from app.config import settings
from app.core.readers.pysd_model_reader import PySDModelReader
from app.exceptions import ModelParseException
from app.schemas.models import ModelInformationSchema, ModelVariableSchema
from app.schemas.validations import ValidationResultSchema


async def upload_mdl_file(file: UploadFile, session_id: str) -> UploadModelResponse:
    """
    Save uploaded .mdl file and read model structure.

    Args:
        file: The uploaded .mdl file.
        session_id: The current session ID for organizing uploads.

    Returns:
        UploadModelResponse: Contains validation results and extracted model
            information.

    Raises:
        ModelParseException: If the file cannot be parsed.
    """
    uploads_dir = settings.TEMP_DIR / session_id / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    file_path = uploads_dir / file.filename
    await file.seek(0)
    content = await file.read()
    file_path.write_bytes(content)

    try:
        reader = PySDModelReader(file_path)
        info = reader.read()
    except Exception as e:
        raise ModelParseException(
            filename=file.filename or "unknown",
            reason=str(e),
        )

    model_info = ModelInformationSchema(
        source_file=info.source_file,
        format=info.format,
        stocks=[ModelVariableSchema(**v.model_dump()) for v in info.stocks],
        flows=[ModelVariableSchema(**v.model_dump()) for v in info.flows],
        parameters=[ModelVariableSchema(**v.model_dump()) for v in info.parameters],
        auxiliaries=[ModelVariableSchema(**v.model_dump()) for v in info.auxiliaries],
    )

    validation = ValidationResultSchema(
        is_valid=True,
        format="mdl",
        checks_passed=["File validation passed", "Model parsed successfully"],
        metadata={"parser": "pysd"},
    )

    return UploadModelResponse(validation=validation, model_info=model_info)
