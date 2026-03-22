from fastapi import File, UploadFile

from app.config import settings
from app.core.validation.validators import MdlFileValidator


async def validate_mdl_file(file: UploadFile = File(...)) -> UploadFile:
    """
    Validates an uploaded .mdl file.

    Args:
        file (UploadFile): The file to validate.

    Returns:
        UploadFile: The validated file.
    """
    validator = MdlFileValidator(max_size=settings.MAX_MODEL_FILE_SIZE)
    await validator.validate(file)
    return file
