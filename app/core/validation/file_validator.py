from abc import ABC, abstractmethod
from typing import Any

from fastapi import UploadFile

from app.exceptions import FileValidationError


class FileValidator(ABC):
    """Abstract base class for file validators."""

    def __init__(
        self,
        max_size: int,
        allowed_extensions: list[str],
    ):
        self.max_size = max_size
        self.allowed_extensions = [ext.lower() for ext in allowed_extensions]

    async def validate(self, file: UploadFile) -> dict[str, Any]:
        """
        Run all validation checks on an uploaded file.

        Args:
            file (UploadFile): The file object to validate, provided by FastAPI.

        Returns:
            dict[str, Any]: Basic information and content of the validated file.
        """
        await file.seek(0)
        file_info = await self._get_file_info(file)

        self._validate_size(file_info)
        self._validate_extension(file_info)
        await self._validate_content(file_info)

        await file.seek(0)
        return file_info

    async def _get_file_info(self, file: UploadFile) -> dict[str, Any]:
        """
        Read file bytes and extract basic info like size and extension.

        Args:
            file (UploadFile): The uploaded file object.

        Returns:
            dict[str, Any]: Dictionary containing filename, size, extension, and content.
        """
        content = await file.read()
        await file.seek(0)

        import os

        ext = os.path.splitext(file.filename or "")[1].lower()

        return {
            "filename": file.filename,
            "size": len(content),
            "extension": ext,
            "content": content,
        }

    def _validate_size(self, file_info: dict[str, Any]) -> None:
        """
        Check the file is not empty and within size limits.

        Args:
            file_info (dict[str, Any]): Dictionary containing file size.

        Returns:
            None: Returns nothing if successful.

        Raises:
            FileValidationError: If the file is 0 bytes or exceeds max_size.
        """
        if file_info["size"] == 0:
            raise FileValidationError("File is empty (0 bytes).", "EMPTY_FILE")

        if file_info["size"] > self.max_size:
            size_mb = file_info["size"] / (1024 * 1024)
            limit_mb = self.max_size / (1024 * 1024)
            raise FileValidationError(
                f"File size ({size_mb:.1f} MB) exceeds the {limit_mb:.0f} MB limit.",
                "FILE_TOO_LARGE",
            )

    def _validate_extension(self, file_info: dict[str, Any]) -> None:
        """
        Check the file extension is allowed.

        Args:
            file_info (dict[str, Any]): Dictionary containing file extension.

        Returns:
            None: Returns nothing if successful.

        Raises:
            FileValidationError: If the file has no extension or is not allowed.
        """
        ext = file_info["extension"]
        if not ext:
            raise FileValidationError("File has no extension.", "NO_EXTENSION")

        if ext not in self.allowed_extensions:
            raise FileValidationError(
                f"Extension '{ext}' is not allowed. "
                f"Allowed: {', '.join(self.allowed_extensions)}",
                "INVALID_EXTENSION",
            )

    @abstractmethod
    async def _validate_content(self, file_info: dict[str, Any]) -> None:
        """
        Format-specific content validation (implemented by subclasses).

        Args:
            file_info (dict[str, Any]): Dictionary containing file details and raw bytes.

        Returns:
            None: Modifies file_info in-place and returning nothing if successful.
        """
        ...
