class FileValidationError(Exception):
    """Raised when an uploaded file fails validation."""

    def __init__(self, message: str, code: str = "VALIDATION_ERROR"):
        self.message = message
        self.code = code
        super().__init__(self.message)


class ModelParseException(Exception):
    """Raised when a model file cannot be parsed or processed."""

    def __init__(self, filename: str, reason: str):
        self.filename = filename
        self.reason = reason
        super().__init__(f"Failed to parse '{filename}': {reason}")


class SimulationException(Exception):
    """Raised when a simulation fails to execute."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Simulation failed: {reason}")
