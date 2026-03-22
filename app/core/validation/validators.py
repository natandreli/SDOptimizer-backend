from __future__ import annotations

import re
from typing import Any

from app.core.validation.file_validator import FileValidator
from app.exceptions import FileValidationError
from app.schemas.validations import ValidationResultSchema


class MdlFileValidator(FileValidator):
    """
    Validator for .mdl (Vensim) model files.

    Checks:
      1. File is readable as text (not binary)
      2. Low ratio of non-printable characters
      3. Required Vensim keywords present (TIME STEP, FINAL TIME, INITIAL TIME)
      4. Valid equation structure (pipe separators + equation count)
      5. No code-injection patterns
    """

    REQUIRED_KEYWORDS = ["TIME STEP", "FINAL TIME", "INITIAL TIME"]

    CONFIDENCE_KEYWORDS = [
        "INTEG",
        "SMOOTH",
        "DELAY",
        "PULSE",
        "STEP",
        "MIN",
        "MAX",
        "ABS",
        "EXP",
        "LN",
        "SQRT",
        "IF THEN ELSE",
        "WITH LOOKUP",
    ]

    SUSPICIOUS_PATTERNS = [
        r"import\s+os",
        r"import\s+subprocess",
        r"__import__",
        r"eval\s*\(",
        r"exec\s*\(",
        r"open\s*\(",
        r"subprocess\.",
        r"os\.system",
        r"os\.popen",
        r"shutil\.",
        r"<script",
        r"javascript:",
        r"cmd\.exe",
        r"/bin/sh",
        r"/bin/bash",
    ]

    MAX_BINARY_RATIO = 0.02  # 2 %

    def __init__(self, max_size: int = 50 * 1024 * 1024):
        super().__init__(
            max_size=max_size,
            allowed_extensions=[".mdl"],
        )

    async def _validate_content(self, file_info: dict[str, Any]) -> None:
        """
        Run all mdl-specific checks on the raw file bytes.

        Raises FileValidationError on the first failing check.
        Stores a ValidationResult in file_info["validation_result"].

        Args:
            file_info (dict[str, Any]): Dictionary containing file contents and metadata.

        Returns:
            None: Modifies file_info in-place with the validation result.
        """
        raw: bytes = file_info["content"]

        passed: list[str] = []
        meta: dict[str, Any] = {}

        text, encoding = self._decode_text(raw)
        if text is None:
            raise FileValidationError(
                "The file cannot be decoded as text. Vensim .mdl files are plain text.",
                "NOT_TEXT_FILE",
            )
        passed.append(f"Readable text file (encoding: {encoding})")
        meta["encoding"] = encoding
        meta["char_count"] = len(text)

        binary_ratio = self._binary_ratio(raw)
        if binary_ratio > self.MAX_BINARY_RATIO:
            raise FileValidationError(
                f"File contains {binary_ratio * 100:.1f}% non-printable characters. "
                "A Vensim .mdl is pure text.",
                "HIGH_BINARY_RATIO",
            )
        passed.append(
            f"Content is plain text (binary ratio: {binary_ratio * 100:.2f}%)"
        )

        text_upper = text.upper()
        missing = [kw for kw in self.REQUIRED_KEYWORDS if kw not in text_upper]
        if missing:
            raise FileValidationError(
                f"Missing required Vensim keywords: {', '.join(missing)}",
                "MISSING_VENSIM_KEYWORDS",
            )
        passed.append("Required Vensim keywords found")

        found_confidence = [kw for kw in self.CONFIDENCE_KEYWORDS if kw in text_upper]
        meta["confidence_keywords"] = found_confidence
        if found_confidence:
            passed.append(f"Additional SD keywords: {', '.join(found_confidence[:5])}")

        eq_check = self._check_equation_structure(text)
        if not eq_check["valid"]:
            raise FileValidationError(eq_check["detail"], "INVALID_EQUATION_STRUCTURE")
        passed.append(
            f"Valid equation structure ({eq_check['equation_count']} equations)"
        )
        meta.update(eq_check)

        suspicious = self._check_suspicious_patterns(text)
        if suspicious:
            raise FileValidationError(
                f"Suspicious patterns found: {', '.join(suspicious)}",
                "SUSPICIOUS_CONTENT",
            )
        passed.append("No code-injection patterns detected")

        file_info["validation_result"] = ValidationResultSchema(
            is_valid=True,
            format="mdl",
            checks_passed=passed,
            metadata=meta,
        )

    @staticmethod
    def _decode_text(raw: bytes) -> tuple[str | None, str]:
        """
        Try to decode bytes as text (UTF-8 → latin-1 → cp1252).

        Args:
            raw (bytes): The raw file bytes to decode.

        Returns:
            tuple[str | None, str]: A tuple containing the decoded text and the encoding used,
                or (None, "") if decoding failed.
        """
        if raw[:3] == b"\xef\xbb\xbf":
            raw = raw[3:]  # strip UTF-8 BOM
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                return raw.decode(enc), enc
            except (UnicodeDecodeError, ValueError):
                pass
        return None, ""

    @staticmethod
    def _binary_ratio(raw: bytes) -> float:
        """
        Calculate the ratio of non-printable bytes in the file.

        Args:
            raw (bytes): The raw file bytes.

        Returns:
            float: The ratio of non-printable characters (0.0 to 1.0).
        """
        if not raw:
            return 0.0
        printable = sum(1 for b in raw if 0x20 <= b <= 0x7E or b in (0x09, 0x0A, 0x0D))
        return 1.0 - (printable / len(raw))

    @staticmethod
    def _check_equation_structure(text: str) -> dict:
        """
        Verify Vensim equation structure (pipes + equations).

        Args:
            text (str): The full decoded text of the model file.

        Returns:
            dict: A dictionary containing the validation result, details, and counts
                of key structural elements (equations, pipes, etc.).
        """
        pipe_count = text.count("|")
        tilde_count = text.count("~")
        integ_count = len(re.findall(r"\bINTEG\s*\(", text, re.IGNORECASE))
        equation_count = len(re.findall(r"^.+\s*=\s*.+", text, re.MULTILINE))

        if pipe_count < 2 or equation_count < 3:
            return {
                "valid": False,
                "detail": (
                    "Invalid Vensim model: not enough equations or pipe separators."
                ),
                "equation_count": equation_count,
            }

        return {
            "valid": True,
            "detail": "OK",
            "equation_count": equation_count,
            "integ_count": integ_count,
            "pipe_count": pipe_count,
            "tilde_count": tilde_count,
        }

    @classmethod
    def _check_suspicious_patterns(cls, text: str) -> list[str]:
        """
        Find code-injection patterns that should not appear in a .mdl file.

        Args:
            text (str): The full decoded text of the model file.

        Returns:
            list[str]: A list of suspicious patterns found in the text, empty if none.
        """
        return [
            pat
            for pat in cls.SUSPICIOUS_PATTERNS
            if re.search(pat, text, re.IGNORECASE)
        ]
