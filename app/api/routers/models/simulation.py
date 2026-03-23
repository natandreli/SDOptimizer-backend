"""Simulation endpoint for executing dynamic model simulations."""

from fastapi import APIRouter, HTTPException, Request

from app.api.routers.models.response_schemas import SimulationResponse
from app.core.operations.models import simulate_model
from app.exceptions import ModelParseException, SimulationException
from app.schemas.simulation import SimulationConfigSchema

router = APIRouter()


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
