from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Any

import pysd

from app.core.patching import PySDFunctionPatcher

from app.schemas.models import ModelInformationSchema, ModelVariableSchema


class PySDModelReader:
    """
    Reads Vensim .mdl files using PySD library.

    PySD is the standard open-source library for System Dynamics in Python.
    It provides robust parsing and classification of Vensim models.

    Features:
    - Official Vensim parser (no fragile regex)
    - Extracts variables with full metadata
    - Automatic classification (stock/flow/parameter/auxiliary)
    - Dependency graph generation
    - Low-level model compilation (LIM)
    """

    def __init__(self, filepath: str | Path):
        """
        Initialize PySD model reader.

        Args:
            filepath: Path to the .mdl file

        Raises:
            FileNotFoundError: If file does not exist
        """
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")

    def read(self) -> ModelInformationSchema:
        """
        Parse .mdl file and extract model structure.

        Uses PySD to load and parse the model. Converts internal PySD
        representation to ModelInformationSchema for API compatibility.

        Returns:
            ModelInformationSchema: Container with all model variables

        Raises:
            Exception: If PySD fails to parse the file
        """
        PySDFunctionPatcher.apply()

        # Load model using PySD
        model = pysd.read_vensim(str(self.filepath))

        # Initialize schema
        info = ModelInformationSchema(
            source_file=str(self.filepath),
            format="mdl",
            metadata={
                "parser": "pysd",
                "pysd_version": pysd.__version__,
            },
        )

        doc = model.doc
        if doc is None or doc.empty:
            return info

        component_module = model.components._components
        flow_py_names = self._detect_flow_py_names(component_module)

        for _, row in doc.iterrows():
            real_name = str(row.get("Real Name", "")).strip()
            py_name = str(row.get("Py Name", "")).strip()

            if not real_name or not py_name:
                continue

            if py_name in {
                "time",
                "initial_time",
                "final_time",
                "time_step",
                "saveper",
            }:
                continue

            unit = self._to_str(row.get("Units"))
            description = self._to_str(row.get("Comment"))

            definition = self._get_element_definition(component_module, py_name)

            var_type = self._classify_element(
                comp_type=self._to_str(row.get("Type")),
                py_name=py_name,
                flow_py_names=flow_py_names,
            )

            var = ModelVariableSchema(
                name=real_name,
                type=var_type,
                equation=definition or "",
                unit=unit,
                description=description,
            )

            if var_type == "parameter" and definition:
                try:
                    var.initial_value = float(definition)
                except ValueError:
                    pass

            if var_type == "stock":
                info.stocks.append(var)
            elif var_type == "flow":
                info.flows.append(var)
            elif var_type == "parameter":
                info.parameters.append(var)
            else:
                info.auxiliaries.append(var)

            info.raw_equations[real_name] = definition or ""

        info.metadata["detected_flow_count"] = len(flow_py_names)

        return info

    @staticmethod
    def _to_str(value: Any) -> str:
        if value is None:
            return ""
        value_str = str(value).strip()
        return "" if value_str.lower() == "nan" else value_str

    @staticmethod
    def _get_element_definition(component_module: Any, py_name: str) -> str | None:
        """
        Extract equation from the generated PySD component function source.

        Args:
            component_module: Compiled module in model.components._components
            py_name: Python component name

        Returns:
            Equation string or None if not available
        """
        try:
            func = getattr(component_module, py_name)
            source = inspect.getsource(func)
            for line in source.splitlines():
                stripped = line.strip()
                if stripped.startswith("return "):
                    return stripped.replace("return ", "", 1).strip()
            return None
        except Exception:
            return None

    @staticmethod
    def _detect_flow_py_names(component_module: Any) -> set[str]:
        """
        Detect flow variables by inspecting Integ ddt lambdas.

        Example extracted text: "lambda: nac_conejos() - muert_conejos()"
        """
        flow_names: set[str] = set()
        pattern = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")

        for name in dir(component_module):
            if not name.startswith("_integ_"):
                continue

            try:
                integ_obj = getattr(component_module, name)
                source = inspect.getsource(integ_obj.ddt)
                for match in pattern.findall(source):
                    if match not in {"lambda"} and not match.startswith("_integ_"):
                        flow_names.add(match)
            except Exception:
                continue

        return flow_names

    @staticmethod
    def _classify_element(
        comp_type: str,
        py_name: str,
        flow_py_names: set[str],
    ) -> str:
        """
        Classify element type from PySD documentation and inferred flows.

        Args:
            comp_type: Value from model.doc['Type']
            py_name: Value from model.doc['Py Name']
            flow_py_names: Flows inferred from INTEG derivatives

        Returns:
            Type: "stock", "flow", "parameter", or "auxiliary"
        """
        comp_type_lower = comp_type.lower()
        if comp_type_lower == "stateful":
            return "stock"
        if comp_type_lower == "constant":
            return "parameter"
        if py_name in flow_py_names:
            return "flow"
        return "auxiliary"
