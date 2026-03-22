"""
PySD Function Patcher
=====================

Monkey-patches ``pysd.py_backend.functions.not_implemented_function`` so that
unsupported Vensim functions (e.g. REINITIAL) are handled with fallback
implementations instead of raising ``NotImplementedError``.

Usage::

    from app.core.patching import PySDFunctionPatcher

    # Register additional fallbacks (optional)
    PySDFunctionPatcher.register("my_custom_func", lambda name, val: val)

    # Apply the patch (idempotent)
    PySDFunctionPatcher.apply()

    # Now pysd.read_vensim(...) will use the fallback for any registered
    # function instead of crashing.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from pysd.py_backend import functions

logger = logging.getLogger(__name__)

# Type alias for fallback callables.
# They receive the same *args that not_implemented_function would get:
#   args[0] = Vensim function name (str)
#   args[1:] = the values passed to that function
FallbackFn = Callable[..., Any]


class PySDFunctionPatcher:
    """
    Registry of fallback implementations for Vensim functions that PySD
    marks as ``not_implemented_function``.

    Keeps the original ``not_implemented_function`` reference so it can
    still raise for truly unsupported functions that have no fallback.
    """

    _registry: dict[str, FallbackFn] = {}
    _original_fn: Callable[..., Any] | None = None
    _applied: bool = False

    # ------------------------------------------------------------------
    # Built-in fallbacks
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_reinitial(*args: Any) -> Any:
        """
        Fallback for Vensim's REINITIAL function.

        ``REINITIAL(expr)`` returns the value of *expr* evaluated at the
        initial time.  As a safe approximation we simply return the
        current value — which is exact at initialization time and a
        reasonable default otherwise.

        PySD translates calls as::

            not_implemented_function("reinitial", <value>)

        so ``args`` is ``("reinitial", <value>)``.
        """
        # args = ("reinitial", value)
        return args[1] if len(args) > 1 else 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def register(cls, vensim_name: str, fallback: FallbackFn) -> None:
        """
        Register a fallback for a Vensim function that PySD does not
        implement.

        Parameters
        ----------
        vensim_name:
            Lowercase name of the Vensim function (e.g. ``"reinitial"``).
        fallback:
            Callable that will receive the same ``*args`` as
            ``not_implemented_function`` — where ``args[0]`` is the
            function name and ``args[1:]`` are the values.
        """
        cls._registry[vensim_name.lower()] = fallback
        logger.debug("Registered fallback for Vensim function '%s'", vensim_name)

    @classmethod
    def apply(cls) -> None:
        """
        Replace ``pysd.py_backend.functions.not_implemented_function``
        with a dispatcher that checks the registry before raising.

        This method is **idempotent** — calling it more than once is safe.
        """
        if cls._applied:
            return

        # Save the original so we can delegate for unknown functions
        cls._original_fn = functions.not_implemented_function

        # Register built-in fallbacks
        if "reinitial" not in cls._registry:
            cls.register("reinitial", cls._fallback_reinitial)

        # Patch
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
        Replacement for ``functions.not_implemented_function``.

        Looks up the function name (``args[0]``) in the registry.
        If a fallback exists, it is called.  Otherwise, the original
        ``NotImplementedError`` is raised.
        """
        func_name = str(args[0]).lower() if args else ""

        fallback = cls._registry.get(func_name)
        if fallback is not None:
            logger.debug(
                "Using fallback for not-implemented Vensim function '%s'",
                func_name,
            )
            return fallback(*args)

        # No fallback — delegate to the original behaviour
        if cls._original_fn is not None:
            return cls._original_fn(*args)

        raise NotImplementedError(f"Not implemented function '{func_name}'")
