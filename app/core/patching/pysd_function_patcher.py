from __future__ import annotations

import logging
from typing import Any, Callable

from pysd.py_backend import functions

logger = logging.getLogger(__name__)


FallbackFn = Callable[..., Any]


class PySDFunctionPatcher:
    _registry: dict[str, FallbackFn] = {}
    _original_fn: Callable[..., Any] | None = None
    _applied: bool = False

    @staticmethod
    def _fallback_reinitial(*args: Any) -> Any:
        """
        Fallback for Vensim's REINITIAL function.

        Args:
            *args: Variable arguments where args[0] is the function name and args[1] is the value.

        Returns:
            The value from args[1] if present, otherwise 0.
        """
        return args[1] if len(args) > 1 else 0

    @classmethod
    def register(cls, vensim_name: str, fallback: FallbackFn) -> None:
        """
        Register a fallback function for a Vensim function.

        Args:
            vensim_name: The name of the Vensim function to register.
            fallback: The fallback function to use as replacement.
        """
        cls._registry[vensim_name.lower()] = fallback
        logger.debug("Registered fallback for Vensim function '%s'", vensim_name)

    @classmethod
    def apply(cls) -> None:
        """
        Apply the patch to pysd.py_backend.functions.not_implemented_function.
            - Replaces the target function with a dispatcher that checks for registered fallbacks.
            - Ensures idempotency by only applying once and storing the original function.
        """
        if cls._applied:
            return

        cls._original_fn = functions.not_implemented_function

        if "reinitial" not in cls._registry:
            cls.register("reinitial", cls._fallback_reinitial)

        functions.not_implemented_function = cls._patched_not_implemented
        cls._applied = True
        logger.info(
            "PySD not_implemented_function patched — %d fallback(s) registered: %s",
            len(cls._registry),
            ", ".join(cls._registry.keys()),
        )

    @classmethod
    def _patched_not_implemented(cls, *args: Any) -> Any:
        """
        Replacement dispatcher for functions.not_implemented_function.

        Args:
            *args: Variable arguments where args[0] is the function name.

        Returns:
            The result from the registered fallback function if one exists,
            or delegates to the original function.

        Raises:
            NotImplementedError: If no fallback is found and original function also fails.
        """
        func_name = str(args[0]).lower() if args else ""

        fallback = cls._registry.get(func_name)
        if fallback is not None:
            logger.debug(
                "Using fallback for not-implemented Vensim function '%s'",
                func_name,
            )
            return fallback(*args)

        if cls._original_fn is not None:
            return cls._original_fn(*args)

        raise NotImplementedError(f"Not implemented function '{func_name}'")
