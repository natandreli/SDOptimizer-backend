from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile

from app.api.dependencies.file_validation import validate_mdl_file
from app.api.routers.models.response_schemas import UploadModelResponse
from app.api.routers.models.simulation import router as simulation_router
from app.core.operations.models import upload_mdl_file
from app.exceptions import FileValidationError, ModelParseException

router = APIRouter(
    prefix="/models",
    tags=["Models"],
)

# Include simulation sub-router
router.include_router(simulation_router)


@router.post(
    "/upload/mdl",
    response_model=UploadModelResponse,
    description="Upload and validate a .mdl (Vensim) model file.",
)
async def handle_upload_mdl_file(
    request: Request,
    file: UploadFile = Depends(validate_mdl_file),
):
    try:
        return await upload_mdl_file(file, session_id=request.state.session_id)
    except FileValidationError as e:
        raise HTTPException(
            status_code=400,
            detail=f"File validation failed: {e.message}",
        )
    except ModelParseException as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to parse '{e.filename}': {e.reason}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error uploading file: {str(e)}",
        )
