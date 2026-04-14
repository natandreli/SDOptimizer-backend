from __future__ import annotations

import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import UploadFile

from app.api.routers.models.response_schemas import (
    GetModelResponse,
    UploadModelResponse,
)
from app.config import settings
from app.core.agent.e_greedy_agent import EGreedytAgent
from app.core.optimizer.model_optimizer import ModelOptimizer
from app.core.readers.pysd_model_reader import PySDModelReader
from app.core.simulator.pysd_simulator import PySDSimulator
from app.exceptions import ModelParseException, SimulationException
from app.schemas.models import ModelSchema, ModelVariableSchema
from app.schemas.optimizer import (
    OptimizationHistorySchema,
    OptimizationResponse,
    OptimizationResultSchema,
)
from app.schemas.simulation import SimulationConfigSchema, SimulationResultSchema


def _get_models_dir(session_id: str | None) -> Path:
    """
    Get the directory for a specific session.

    Args:
        session_id: Session identifier to organize uploaded models. If None, returns a default directory.

    Returns:
        Path: Directory path for the session's models
    """
    if session_id:
        return settings.TEMP_DIR / session_id / "uploads"
    return settings.TEMP_DIR / "default" / "uploads"


async def get_all_models(session_id: str) -> list[GetModelResponse]:
    """
    Retrieve all uploaded models for the current session.

    Args:
        session_id: The current session ID.

    Returns:
        list[GetModelResponse]: A list of uploaded models for the session.
    """
    uploads_dir = _get_models_dir(session_id)
    if not uploads_dir.exists():
        return []

    models: list[GetModelResponse] = []
    seen_file_names: set[str] = set()

    for model_dir in sorted(
        (p for p in uploads_dir.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        mdl_files = list(model_dir.glob("*.mdl"))
        if not mdl_files:
            continue

        file_path = mdl_files[0]
        if file_path.name in seen_file_names:
            continue
        seen_file_names.add(file_path.name)

        try:
            reader = PySDModelReader(file_path)
            info = reader.read()

            model = ModelSchema(
                file_name=file_path.name,
                uploaded_at=datetime.fromtimestamp(
                    file_path.stat().st_mtime,
                    tz=timezone.utc,
                ).isoformat(),
                parsed_with="pysd",
                format=info.format,
                stocks=[ModelVariableSchema(**v.model_dump()) for v in info.stocks],
                flows=[ModelVariableSchema(**v.model_dump()) for v in info.flows],
                parameters=[
                    ModelVariableSchema(**v.model_dump()) for v in info.parameters
                ],
                auxiliaries=[
                    ModelVariableSchema(**v.model_dump()) for v in info.auxiliaries
                ],
            )

            models.append(
                GetModelResponse(
                    model_id=model_dir.name,
                    model=model,
                )
            )
        except Exception:
            models.append(
                GetModelResponse(
                    model_id=model_dir.name,
                    model=None,
                )
            )

    return models


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
    uploads_dir = settings.TEMP_DIR / session_id / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    if not file.filename:
        raise ModelParseException(
            filename="unknown",
            reason="Missing filename.",
        )

    for existing_dir in uploads_dir.iterdir():
        if existing_dir.is_dir() and (existing_dir / file.filename).exists():
            model_id = existing_dir.name
            break

    await file.seek(0)
    content = await file.read()

    try:
        with tempfile.TemporaryDirectory(prefix="sdoptimizer-parse-") as tmp_dir:
            parse_tmp_file = Path(tmp_dir) / file.filename
            parse_tmp_file.write_bytes(content)
            reader = PySDModelReader(parse_tmp_file)
            info = reader.read()
    except Exception as e:
        raise ModelParseException(
            filename=file.filename,
            reason=str(e),
        )

    model_dir = uploads_dir / model_id
    model_dir.mkdir(parents=True, exist_ok=True)

    file_path = model_dir / file.filename
    file_path.write_bytes(content)

    model = ModelSchema(
        file_name=file.filename,
        uploaded_at=datetime.now(timezone.utc).isoformat(),
        parsed_with="pysd",
        format=info.format,
        stocks=[ModelVariableSchema(**v.model_dump()) for v in info.stocks],
        flows=[ModelVariableSchema(**v.model_dump()) for v in info.flows],
        parameters=[ModelVariableSchema(**v.model_dump()) for v in info.parameters],
        auxiliaries=[ModelVariableSchema(**v.model_dump()) for v in info.auxiliaries],
    )

    return UploadModelResponse(
        model_id=model_id,
        model=model,
    )


def delete_model(model_id: str, session_id: str) -> None:
    """
    Delete a user-generated model directory.

    Args:
        model_id: The unique ID of the model to delete.
        session_id: The current session ID for locating the model.

    Raises:
        ModelParseException: If the model cannot be found.
    """
    model_dir = settings.TEMP_DIR / session_id / "uploads" / model_id

    if not model_dir.exists() or not model_dir.is_dir():
        raise ModelParseException(
            filename=model_id,
            reason="Model not found. Cannot delete.",
        )

    shutil.rmtree(model_dir)


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
        simulator = PySDSimulator(pysd_model, config)
        result = simulator.simulate()
    except SimulationException:
        raise
    except Exception as e:
        raise SimulationException(reason=str(e))

    return result


async def optimize_model(
    session_id: str, model_id: str, config
) -> OptimizationResponse:
    """
    Execute ε-greedy multi-armed bandit optimization over a PySD model.

    Args:
        session_id: Unique session identifier used for isolating user data.
        model_id: Identifier of the uploaded model to optimize.
        config: Optimization configuration object

    returns: Dictionary containing best parameters, best score, and optimization history.

    Raises:
        ModelParseException: If the model cannot be loaded.
        ValueError: If configuration or objective function is invalid.
        SimulationException: If simulation execution fails.
    """

    model_path, parameters = load_model(session_id, model_id)

    wrapper = PySDParser(
        model_path=model_path,
        parameters=parameters,
    )

    def objective_fn(df):
        if config.target_variable not in df.columns:
            raise ValueError(
                f"Variable '{config.target_variable}' not found in simulation results."
            )

        if config.statistic == "final":
            return float(df[config.target_variable].iloc[-1])
        elif config.statistic == "mean":
            return float(df[config.target_variable].mean())
        elif config.statistic == "max":
            return float(df[config.target_variable].max())
        else:
            raise ValueError(f"Unknown statistic: {config.statistic}")

    action_shape = (3,) * len(config.parameter_names)

    agent = EGreedytAgent(
        action_shape=action_shape,
        epsilon=config.epsilon,
    )

    optimizer = ModelOptimizer(
        wrapper=wrapper,
        parameter_names=config.parameter_names,
        initial_values=config.initial_values,
        bounds=config.bounds,
        rho_factors=config.rho_factors,
        agent=agent,
        objective_fn=objective_fn,
        max_runs=config.max_runs,
    )

    best_params, best_score = optimizer.optimize()
    history = optimizer.get_history()

    return OptimizationResponse(
        result=OptimizationResultSchema(
            best_parameters=dict(zip(config.parameter_names, best_params)),
            best_score=best_score,
            history=OptimizationHistorySchema(**history),
        )
    )
