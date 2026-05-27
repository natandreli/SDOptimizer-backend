from __future__ import annotations

import shutil
import tempfile
import time
import uuid
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

from fastapi import UploadFile

from app.schemas.models import (
    GetModelResponse,
    ModelSchema,
    ModelVariableSchema,
    UploadModelResponse,
)
from app.config import settings
from app.core.agent.e_greedy_agent import EGreedyAgent
from app.core.optimizer.model_optimizer import ModelOptimizer
from app.core.readers.pysd_model_reader import PySDModelReader
from app.core.readers.pysd_parser import PySDParser
from app.core.simulator.pysd_simulator import PySDSimulator
from app.core.utils.model_loader import load_model
from app.exceptions import ModelParseException, SimulationException
from app.schemas.optimizer import (
    OptimizationConfigSchema,
    OptimizationConfigSummarySchema,
    OptimizationDefaultsSchema,
    OptimizationHistorySchema,
    OptimizationOptionsSchema,
    OptimizationParameterOptionSchema,
    OptimizationResultSchema,
    ParameterChangeSchema,
)
from app.schemas.simulation import (
    SimulationConfigSchema,
    SimulationDefaultsSchema,
    SimulationOptionsSchema,
    SimulationParameterOptionSchema,
    SimulationResultSchema,
)


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

        # FAST PATH: Check if we have cached metadata
        info_path = model_dir / "info.json"
        if info_path.exists():
            try:
                info = ModelSchema.model_validate_json(info_path.read_text())
                models.append(
                    GetModelResponse(
                        model_id=model_dir.name,
                        model=info,
                    )
                )
                continue
            except Exception:
                pass

        #  Parse model and cache it
        try:
            reader = PySDModelReader(file_path)
            info, _ = reader.read()

            # Cache metadata for next time
            info_path.write_text(info.model_dump_json())

            models.append(
                GetModelResponse(
                    model_id=model_dir.name,
                    model=info,
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
            info, _ = reader.read()
            
            model_dir = uploads_dir / model_id
            model_dir.mkdir(parents=True, exist_ok=True)

            file_path = model_dir / file.filename
            file_path.write_bytes(content)

            py_tmp_file = parse_tmp_file.with_suffix(".py")
            if py_tmp_file.exists():
                shutil.copy(py_tmp_file, model_dir / py_tmp_file.name)

            info.uploaded_at = datetime.now(timezone.utc).isoformat()
            info_path = model_dir / "info.json"
            info_path.write_text(info.model_dump_json())

    except Exception as e:
        raise ModelParseException(
            filename=file.filename,
            reason=str(e),
        )

    return UploadModelResponse(
        model_id=model_id,
        model=info,
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

    # Resolve total_time / final_time from the payload, falling back to model defaults
    resolved_total_time = config.final_time if config.final_time is not None else config.total_time
    if resolved_total_time is None:
        try:
            resolved_total_time = float(pysd_model.components.final_time())
        except Exception:
            resolved_total_time = 100.0
    config.total_time = resolved_total_time

    # Resolve dt from the payload, falling back to model defaults
    if config.dt is None:
        try:
            config.dt = float(pysd_model.components.time_step())
        except Exception:
            config.dt = 0.25

    start_time = time.perf_counter()
    result = None
    compiler_success = False

    py_files = list(model_dir.glob("*.py"))
    if py_files:
        try:
            from app.core.compiler.vector_compiler import VectorModelCompiler
            compiler = VectorModelCompiler(py_files[0], pysd_model)
            compiler.compile()
            
            # Determine output recording timestamps
            total_time = config.total_time
            dt = config.dt
            if total_time / dt > 1000:
                steps_per_output = max(1, int((total_time / 1000) / dt))
                output_step = steps_per_output * dt
            else:
                output_step = dt
            return_timestamps = np.arange(0, total_time + output_step, output_step)
            
            series = compiler.simulate(
                parameter_overrides=config.parameter_overrides,
                dt=config.dt,
                total_time=config.total_time,
                return_timestamps=return_timestamps,
                return_columns=config.return_columns,
            )
            
            time_list = series.pop("time")
            series["time"] = time_list
            
            parameter_names = set(compiler.constants)
            parameter_series = {
                name: values
                for name, values in series.items()
                if name in parameter_names and name != "time"
            }
            variable_series = {
                name: values
                for name, values in series.items()
                if name not in parameter_names and name != "time"
            }
            
            summary_stats = {}
            for name, values in series.items():
                if name != "time" and values:
                    arr = np.array(values, dtype=float)
                    summary_stats[name] = {
                        "mean": float(np.nanmean(arr)),
                        "min": float(np.nanmin(arr)),
                        "max": float(np.nanmax(arr)),
                        "final": float(arr[-1]),
                        "initial": float(arr[0]),
                    }
                    
            result = SimulationResultSchema(
                time_series=variable_series,
                parameter_series=parameter_series,
                summary_stats=summary_stats,
                steps_executed=int(config.total_time / config.dt) if config.dt and config.total_time else 1,
                config=config,
            )
            compiler_success = True
            print(f"DEBUG: Simulation for model {model_id} executed with Dynamic Vector Compiler (MIT)!")
        except Exception as e:
            print(f"DEBUG: Dynamic Vector Compiler simulation failed: {e}. Falling back to PySD.")

    if not compiler_success:
        try:
            simulator = PySDSimulator(pysd_model, config)
            result = simulator.simulate()
        except SimulationException:
            raise
        except Exception as e:
            raise SimulationException(reason=str(e))
        finally:
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            print(f"DEBUG: Simulation for model {model_id} took {duration_ms:.2f}ms")

    return result


def _suggest_bounds(initial_value: float) -> tuple[float, float]:
    """
    Build suggested bounds around an initial value.

    Args:
        initial_value: The reference value to generate bounds for.

    Returns:
        tuple[float, float]: A tuple containing (lower, upper) bounds.
    """
    if initial_value == 0:
        return (-1.0, 1.0)

    lower = initial_value * 0.5
    upper = initial_value * 1.5
    return (lower, upper) if lower <= upper else (upper, lower)


def get_optimization_options(
    session_id: str, model_id: str
) -> OptimizationOptionsSchema:
    """
    Build optimization configuration options for a loaded model.

    Args:
        session_id: The current session ID.
        model_id: The unique ID of the uploaded model.

    Returns:
        OptimizationOptionsSchema: Configuration options including parameters and target variables.

    Raises:
        ModelParseException: If the model cannot be read.
        SimulationException: If the simulation fails.
    """
    _, info, _ = load_model(session_id, model_id)

    parameters: list[OptimizationParameterOptionSchema] = []
    for parameter in info.parameters:
        initial_value = (
            float(parameter.initial_value)
            if parameter.initial_value is not None
            else 0.0
        )
        parameters.append(
            OptimizationParameterOptionSchema(
                name=parameter.name,
                initial_value=initial_value,
                suggested_bounds=_suggest_bounds(initial_value),
                suggested_rho_factor=0.01,
            )
        )

    target_variables = [
        variable.name for variable in (info.stocks + info.flows + info.auxiliaries)
    ]

    try:
        internal_initial = float(info.raw_equations.get("INITIAL TIME", 0) or 0)
        internal_final = float(info.raw_equations.get("FINAL TIME", 100) or 100)
        internal_dt = float(info.raw_equations.get("TIME STEP", 0.01) or 0.01)
        
        duration = internal_final - internal_initial
        if duration <= 0:
            duration = 100.0
            
        suggested_total_time = duration
        
        # AGGRESSIVE OPTIMIZATION: Target 100 steps for the optimization loop
        # 500 runs * 100 steps = 50,000 operations (Should take ~20-30s)
        actual_steps = duration / internal_dt if internal_dt > 0 else 0
        if actual_steps > 200:
            suggested_dt = duration / 100.0
        else:
            suggested_dt = internal_dt if internal_dt > 0 else (duration / 100.0)
            
    except Exception:
        suggested_total_time = 100.0
        suggested_dt = 0.25

    return OptimizationOptionsSchema(
        parameters=parameters,
        target_variables=target_variables,
        statistics=["final", "mean", "max", "min"],
        directions=["maximize", "minimize"],
        defaults=OptimizationDefaultsSchema(
            dt=suggested_dt,
            total_time=suggested_total_time,
            time_unit=info.time_unit,
        ),
    )


def get_simulation_options(session_id: str, model_id: str) -> SimulationOptionsSchema:
    """
    Build simulation configuration options for a loaded model.

    Args:
        session_id: The current session ID.
        model_id: The unique ID of the uploaded model.

    Returns:
        SimulationOptionsSchema: Configuration options including parameters.

    Raises:
        ModelParseException: If the model cannot be read.
    """
    _, info, _ = load_model(session_id, model_id)

    parameters: list[SimulationParameterOptionSchema] = []
    for parameter in info.parameters:
        initial_value = (
            float(parameter.initial_value)
            if parameter.initial_value is not None
            else 0.0
        )
        parameters.append(
            SimulationParameterOptionSchema(
                name=parameter.name,
                initial_value=initial_value,
            )
        )

    try:
        internal_initial = float(info.raw_equations.get("INITIAL TIME", 0) or 0)
        internal_final = float(info.raw_equations.get("FINAL TIME", 100) or 100)
        internal_dt = float(info.raw_equations.get("TIME STEP", 0.25) or 0.25)

        duration = internal_final - internal_initial
        if duration <= 0:
            duration = 100.0
            
        suggested_total_time = duration
        suggested_dt = internal_dt if internal_dt > 0 else 0.25
    except Exception:
        suggested_total_time = 100.0
        suggested_dt = 0.25

    return SimulationOptionsSchema(
        parameters=parameters,
        defaults=SimulationDefaultsSchema(
            dt=suggested_dt,
            total_time=suggested_total_time,
            time_unit=info.time_unit,
        ),
    )


async def optimize_model(
    session_id: str,
    model_id: str,
    config: OptimizationConfigSchema,
) -> OptimizationResultSchema:
    """
    Execute ε-greedy multi-armed bandit optimization over a PySD model.

    Args:
        session_id: Unique session identifier used for isolating user data.
        model_id: Identifier of the uploaded model to optimize.
        config: Optimization configuration object

    Returns:
        OptimizationResultSchema with best parameters, best score, and optimization history.

    Raises:
        ModelParseException: If the model cannot be loaded.
        ValueError: If configuration or objective function is invalid.
        SimulationException: If simulation execution fails.
    """

    pysd_model_path, info, pysd_model = load_model(session_id, model_id)
    parameters = [p.model_dump() for p in info.parameters]

    wrapper = PySDParser(
        model_path_or_obj=pysd_model,
        parameters=parameters,
    )

    # --- Optimization Time Settings ---
    dt = config.dt
    total_time = config.final_time if config.final_time is not None else config.total_time

    internal_initial = 0.0
    internal_final = 100.0
    internal_dt = 1.0

    try:
        internal_initial = float(info.raw_equations.get("INITIAL TIME", 0) or 0)
        internal_final = float(info.raw_equations.get("FINAL TIME", 100) or 100)
        internal_dt = float(info.raw_equations.get("TIME STEP", 0.25) or 0.25)
    except Exception:
        pass

    duration = internal_final - internal_initial
    if duration <= 0: 
        duration = 100.0

    if total_time is None: 
        total_time = duration

    if dt is None:
        # Preserve the internal timestep for mathematical fidelity.
        # We will optimize performance by reducing return_timestamps (output density) 
        # instead of the mathematical integration step.
        dt = internal_dt if internal_dt > 0 else 1.0

    def objective_fn(df):
        if config.target_variable not in df.columns:
            raise ValueError(
                f"Variable '{config.target_variable}' not found in simulation results."
            )

        if config.statistic == "final":
            value = float(df[config.target_variable].iloc[-1])
        elif config.statistic == "mean":
            value = float(df[config.target_variable].mean())
        elif config.statistic == "max":
            value = float(df[config.target_variable].max())
        elif config.statistic == "min":
            value = float(df[config.target_variable].min())
        else:
            raise ValueError(f"Unknown statistic: {config.statistic}")

        return value if config.direction == "maximize" else -value

    vector_compiler = None
    model_dir = settings.TEMP_DIR / session_id / "uploads" / model_id
    py_files = list(model_dir.glob("*.py"))
    if py_files:
        try:
            from app.core.compiler.vector_compiler import VectorModelCompiler
            vector_compiler = VectorModelCompiler(py_files[0], pysd_model).compile()
            print(f"DEBUG: Optimization model {model_id} successfully compiled with Dynamic Vector Compiler (MIT).")
        except Exception as e:
            print(f"DEBUG: Dynamic Vector Compiler initialization failed: {e}. Falling back to PySD.")

    action_shape = (3,) * len(config.parameter_names)

    agent = EGreedyAgent(
        action_shape=action_shape,
        epsilon=config.epsilon,
    )

    memo_cache = {}
    cache_hits = 0
    cache_misses = 0

    def reward_fn(params: list[float]) -> float:
        nonlocal cache_hits, cache_misses
        # Round parameters to 8 decimals to prevent tiny precision mismatches
        key = tuple(round(p, 8) for p in params)
        if key in memo_cache:
            cache_hits += 1
            return memo_cache[key]

        cache_misses += 1

        if vector_compiler is not None:
            try:
                overrides = dict(zip(config.parameter_names, params))
                run_kwargs = {}
                if config.statistic == "final":
                    run_kwargs["return_timestamps"] = np.array([internal_initial + total_time])
                else:
                    steps = total_time / dt if dt > 0 else 0
                    if steps > 1000:
                        # Subsample output to max 1000 points without modifying the integration `dt`
                        steps_per_output = max(1, round((total_time / 1000) / dt))
                        output_step = steps_per_output * dt
                        run_kwargs["return_timestamps"] = np.arange(internal_initial, internal_initial + total_time + output_step, output_step)

                series = vector_compiler.simulate(
                    parameter_overrides=overrides,
                    dt=dt,
                    total_time=total_time,
                    return_columns=[config.target_variable],
                    **run_kwargs
                )
                vals = series[config.target_variable]
                if not vals:
                    raise ValueError(f"Variable '{config.target_variable}' returned empty trajectory in Vector Compiler.")

                if config.statistic == "final":
                    value = float(vals[-1])
                elif config.statistic == "mean":
                    value = float(np.mean(vals))
                elif config.statistic == "max":
                    value = float(np.max(vals))
                elif config.statistic == "min":
                    value = float(np.min(vals))
                else:
                    raise ValueError(f"Unknown statistic: {config.statistic}")

                score = value if config.direction == "maximize" else -value
                memo_cache[key] = score
                return score
            except Exception as e:
                print(f"DEBUG: Dynamic Vector Compiler simulation error: {e}. Falling back dynamically to PySD for this trial.")

        overrides = dict(zip(config.parameter_names, params))
        run_kwargs = {}
        # TURBO MODE: If only final value is needed, don't collect all intermediate steps
        if config.statistic == "final":
            run_kwargs["return_timestamps"] = [internal_initial + total_time]
        else:
            # Enforce larger return_timestamps interval (target max 1000 points) to avoid DataFrame overhead in PySD
            steps = total_time / dt if dt > 0 else 0
            if steps > 1000:
                steps_per_output = max(1, round((total_time / 1000) / dt))
                output_step = steps_per_output * dt
                run_kwargs["return_timestamps"] = np.arange(internal_initial, internal_initial + total_time + output_step, output_step)

        results = wrapper.run(
            overrides=overrides,
            return_columns=[config.target_variable],
            dt=dt,
            total_time=total_time,
            **run_kwargs
        )
        score = objective_fn(results)
        memo_cache[key] = score
        return score

    # Compute baseline score with initial parameters
    try:
        raw_initial_score = reward_fn(list(config.initial_values))
    except Exception:
        raw_initial_score = 0.0

    optimizer = ModelOptimizer(
        reward_fn=reward_fn,
        agent=agent,
        parameter_names=config.parameter_names,
        initial_values=config.initial_values,
        bounds=config.bounds,
        rho_factors=config.rho_factors,
        max_runs=config.max_runs,
    )

    start_time = time.perf_counter()
    try:
        best_params, best_score = optimizer.optimize()
    except Exception as e:
        raise SimulationException(reason=f"Optimization execution failed: {str(e)}")
    finally:
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        print(f"DEBUG: Optimization for model {model_id} took {duration_ms:.2f}ms")
        print(f"DEBUG: Memoization Cache Stats - Hits: {cache_hits}, Misses: {cache_misses}")

    history = optimizer.get_history()

    # Convert scores back to original scale when minimizing
    initial_score = (
        -raw_initial_score if config.direction == "minimize" else raw_initial_score
    )
    final_best_score = -best_score if config.direction == "minimize" else best_score

    # Compute improvement percentage
    if abs(initial_score) > 1e-12:
        if config.direction == "minimize":
            improvement_pct = (
                (initial_score - final_best_score) / abs(initial_score)
            ) * 100
        else:
            improvement_pct = (
                (final_best_score - initial_score) / abs(initial_score)
            ) * 100
    else:
        improvement_pct = (
            0.0 if abs(final_best_score - initial_score) < 1e-12 else 100.0
        )

    # Build per-parameter change info
    initial_params_dict = dict(zip(config.parameter_names, config.initial_values))
    best_params_dict = dict(zip(config.parameter_names, best_params))

    parameter_changes: dict[str, ParameterChangeSchema] = {}
    for name in config.parameter_names:
        init_val = initial_params_dict[name]
        opt_val = best_params_dict[name]
        if abs(init_val) > 1e-12:
            change_pct = ((opt_val - init_val) / abs(init_val)) * 100
        else:
            change_pct = 0.0 if abs(opt_val - init_val) < 1e-12 else 100.0
        parameter_changes[name] = ParameterChangeSchema(
            initial_value=init_val,
            optimized_value=opt_val,
            change_percentage=change_pct,
        )

    config_summary = OptimizationConfigSummarySchema(
        target_variable=config.target_variable,
        statistic=config.statistic,
        direction=config.direction,
        max_runs=config.max_runs,
        epsilon=config.epsilon,
    )

    steps_per_sim = int(total_time / dt) if dt and total_time else 1
    total_math_steps = config.max_runs * steps_per_sim

    return OptimizationResultSchema(
        best_parameters=best_params_dict,
        best_score=final_best_score,
        history=OptimizationHistorySchema(**history),
        initial_parameters=initial_params_dict,
        initial_score=initial_score,
        improvement_percentage=round(improvement_pct, 4),
        parameter_changes=parameter_changes,
        config_summary=config_summary,
        steps_per_simulation=steps_per_sim,
        total_mathematical_steps=total_math_steps,
    )
