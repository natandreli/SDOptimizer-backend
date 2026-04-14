from app.config import settings
from app.core.readers.pysd_model_reader import PySDModelReader
from app.exceptions import ModelParseException


def load_model(session_id: str, model_id: str) -> tuple[str, list[dict]]:
    """
    Load model file path and parameter metadata for a given model.

    Args:
        session_id: Current session identifier.
        model_id: Unique model identifier.

    Returns:
        Tuple containing:
            - model_path: Path to the .mdl file
            - parameters: List of parameter metadata dictionaries

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

    try:
        reader = PySDModelReader(file_path)
        info = reader.read()
    except Exception as e:
        raise ModelParseException(
            filename=file_path.name,
            reason=f"Failed to read model metadata: {str(e)}",
        )

    parameters = [p.model_dump() for p in info.parameters]

    return str(file_path), parameters
