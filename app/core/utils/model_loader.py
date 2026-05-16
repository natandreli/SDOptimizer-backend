from __future__ import annotations

import pysd

from app.config import settings
from app.core.readers.pysd_model_reader import PySDModelReader
from app.exceptions import ModelParseException
from app.schemas.models import ModelSchema


def load_model(session_id: str, model_id: str) -> tuple[str, ModelSchema, pysd.PySD]:
    """
    Load model file path, metadata and PySD model object.

    Args:
        session_id: Current session identifier.
        model_id: Unique model identifier.

    Returns:
        Tuple containing:
            - model_path: Path to the .mdl file
            - info: Full ModelSchema containing stocks, flows, parameters, etc.
            - pysd_model: The loaded PySD model object.

    Raises:
        ModelParseException: If model or file cannot be found or parsed.
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

    # FAST PATH: Check for cached metadata
    info_path = model_dir / "info.json"
    info = None
    if info_path.exists():
        try:
            info = ModelSchema.model_validate_json(info_path.read_text())
        except Exception:
            pass

    try:
        reader = PySDModelReader(file_path)
        if info:
            # If we have info, we only need the PySD model object
            pysd_model = reader.load()
        else:
            # If no cached info, read and cache it now
            info, pysd_model = reader.read()
            info_path.write_text(info.model_dump_json())
    except Exception as e:
        raise ModelParseException(
            filename=file_path.name,
            reason=f"Failed to load model: {str(e)}",
        )

    return str(file_path), info, pysd_model
