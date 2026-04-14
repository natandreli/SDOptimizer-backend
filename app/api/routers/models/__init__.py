from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile

from app.api.dependencies.file_validation import validate_mdl_file
from app.api.routers.models.response_schemas import (
    GetModelResponse,
    SimulationResponse,
    UploadModelResponse,
)
from app.core.operations.models import (
    delete_model,
    get_all_models,
    optimize_model,
    simulate_model,
    upload_mdl_file,
)
from app.exceptions import FileValidationError, ModelParseException, SimulationException
from app.schemas.optimizer import OptimizationConfigSchema, OptimizationResponse
from app.schemas.simulation import SimulationConfigSchema

router = APIRouter(
    prefix="/models",
    tags=["Models"],
)


@router.get(
    "/all",
    description="Get a list of all uploaded models for the current session.",
    response_model=list[GetModelResponse],
)
async def handle_get_all_models(request: Request):
    try:
        session_id = request.state.session_id
        return await get_all_models(session_id=session_id)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error retrieving models: {str(e)}",
        )


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
            status_code=422,
            detail=f"Failed to parse '{e.filename}': {e.reason}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error uploading file: {str(e)}",
        )


@router.delete(
    "/{model_id}",
    description="Delete a user-generated model.",
    status_code=204,
)
def handle_delete_model(model_id: str, request: Request):
    try:
        session_id = request.state.session_id
        delete_model(model_id, session_id=session_id)
        return None
    except ModelParseException as e:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{e.filename}' not found: {e.reason}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete model: {str(e)}",
        )


@router.post(
    "/{model_id}/simulate",
    response_model=SimulationResponse,
    description=(
        "Run a simulation on a previously uploaded .mdl model. "
        "The model_id is returned by the upload endpoint. "
        "Builds the model dynamically from parsed equations and executes "
        "Euler integration with the given configuration."
    ),
)
async def handle_simulate(
    request: Request,
    model_id: str,
    config: SimulationConfigSchema,
):
    try:
        result = await simulate_model(
            session_id=request.state.session_id,
            model_id=model_id,
            config=config,
        )
        return SimulationResponse(result=result)
    except ModelParseException as e:
        raise HTTPException(
            status_code=404,
            detail=f"Model error: {e.reason}",
        )
    except SimulationException as e:
        raise HTTPException(
            status_code=422,
            detail=f"Simulation failed: {e.reason}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during simulation: {str(e)}",
        )


@router.post(
    "/{model_id}/optimize",
    response_model=OptimizationResponse,
    description=(
        "Run Q-learning optimization on a previously uploaded model. "
        "The optimizer adjusts model parameters to maximize a given objective function."
    ),
)
async def handle_optimize(
    request: Request,
    model_id: str,
    config: OptimizationConfigSchema,
):
    try:
        result = await optimize_model(
            session_id=request.state.session_id,
            model_id=model_id,
            config=config,
        )
        return OptimizationResponse(result=result)

    except ModelParseException as e:
        raise HTTPException(
            status_code=404,
            detail=f"Model error: {e.reason}",
        )
    except SimulationException as e:
        raise HTTPException(
            status_code=422,
            detail=f"Optimization failed: {e.reason}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during optimization: {str(e)}",
        )
