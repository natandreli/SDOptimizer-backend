from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from fastapi import UploadFile

from app.config import settings
from app.core.readers.pysd_model_reader import PySDModelReader
from app.exceptions import ModelParseException, SimulationException
from app.schemas.models import ModelInformationSchema, ModelVariableSchema
from app.schemas.simulation import SimulationConfigSchema, SimulationResultSchema
from app.schemas.validations import ValidationResultSchema

if TYPE_CHECKING:
    from app.api.routers.models.response_schemas import UploadModelResponse


async def upload_mdl_file(file: UploadFile, session_id: str) -> UploadModelResponse:
    """
    Save uploaded .mdl file and read model structure.

    Generates a unique model_id (UUID) and stores the file inside
    a subdirectory named after that ID.

    Args:
        file: The uploaded .mdl file.
        session_id: The current session ID for organizing uploads.

    Returns:
        UploadModelResponse: Contains model_id, validation results,
            and extracted model information.

    Raises:
        ModelParseException: If the file cannot be parsed.
    """
    model_id = uuid.uuid4().hex[:12]

    # Check if same filename was already uploaded — reuse existing model_id
    uploads_root = settings.TEMP_DIR / session_id / "uploads"
    if uploads_root.exists():
        for existing_dir in uploads_root.iterdir():
            if existing_dir.is_dir() and (existing_dir / file.filename).exists():
                model_id = existing_dir.name
                break

    model_dir = uploads_root / model_id
    model_dir.mkdir(parents=True, exist_ok=True)

    file_path = model_dir / file.filename
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

    # Lazy import to avoid circular dependency with routers
    from app.api.routers.models.response_schemas import UploadModelResponse

    return UploadModelResponse(
        model_id=model_id,
        validation=validation,
        model_info=model_info,
    )


async def simulate_model(
    session_id: str,
    model_id: str,
    config: SimulationConfigSchema,
) -> SimulationResultSchema:
    """
    Build and run a simulation from a previously uploaded .mdl file.

    Locates the .mdl file using the model_id, loads it natively with PySD,
    and executes the simulation.

    Args:
        session_id: The current session ID.
        model_id: The unique ID returned when the model was uploaded.
        config: Simulation configuration (dt, total_time, parameter_overrides).

    Returns:
        SimulationResultSchema with time-series data and summary statistics.

    Raises:
        ModelParseException: If the model file cannot be found or parsed.
        SimulationException: If the simulation fails.
    """
    model_dir = settings.TEMP_DIR / session_id / "uploads" / model_id

    if not model_dir.exists():
        raise ModelParseException(
            filename=model_id,
            reason="Model not found. Upload a model first.",
        )

    # Find the .mdl file inside the model directory
    mdl_files = list(model_dir.glob("*.mdl"))
    if not mdl_files:
        raise ModelParseException(
            filename=model_id,
            reason="No .mdl file found in model directory.",
        )

    file_path = mdl_files[0]

    try:
        reader = PySDModelReader(file_path)
        pysd_model = reader.load()
    except Exception as e:
        raise ModelParseException(
            filename=file_path.name,
            reason=str(e),
        )

    try:
        # Lazy import to avoid circular dependency with routers
        from app.core.simulator.pysd_simulator import PySDSimulator

        simulator = PySDSimulator(pysd_model, config)
        result = simulator.simulate()
    except SimulationException:
        raise
    except Exception as e:
        raise SimulationException(reason=str(e))

    return result
