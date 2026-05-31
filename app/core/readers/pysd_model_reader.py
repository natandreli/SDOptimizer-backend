from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Any

import pysd

from app.core.patching import PySDFunctionPatcher
from app.schemas.models import ModelSchema, ModelVariableSchema


class PySDModelReader:
    """
    Reads Vensim .mdl files using PySD library.

    PySD is the standard open-source library for System Dynamics in Python.
    It provides robust parsing and classification of Vensim models.

    Features:
    - Official Vensim parser (no fragile regex)
    - Extracts model variables
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

    def load(self) -> pysd.PySD:
        """
        Load model file and return the PySD model object directly for simulation.
        If a compiled .py version exists, it loads it directly for speed.

        Returns:
            pysd.PySD: The loaded PySD model.
        """
        # If the file is already a .py file, load it directly
        if self.filepath.suffix.lower() == ".py":
            return pysd.load(str(self.filepath))

        # Check if a compiled .py file already exists in the same directory
        py_file = self.filepath.with_suffix(".py")
        if py_file.exists():
            try:
                # IMPORTANT: PySD's load is much faster than read_vensim
                return pysd.load(str(py_file))
            except Exception:
                # If loading .py fails, fallback to read_vensim
                pass

        return pysd.read_vensim(str(self.filepath))

    def read(self) -> tuple[ModelSchema, pysd.PySD]:
        """
        Parse .mdl file and extract model structure.

        Uses PySD to load and parse the model. Converts internal PySD
        representation to ModelSchema for API compatibility.

        Returns:
            tuple[ModelSchema, pysd.PySD]: A tuple containing the ModelSchema
                and the loaded PySD model object.

        Raises:
            Exception: If PySD fails to parse the file
        """
        PySDFunctionPatcher.apply()

        # Load model using optimized load method
        model = self.load()

        # Initialize schema
        info = ModelSchema(
            file_name=self.filepath.name,
            format="mdl",
        )

        doc = model.doc
        if doc is None or doc.empty:
            return info, model

        component_module = model.components._components

        # PRE-EXTRACTION: Get all source code once to avoid slow inspect.getsource calls in loops
        source_map = self._extract_source_map(component_module)
        flow_py_names = self._detect_flow_py_names(source_map)

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
                if py_name == "time_step":
                    info.time_unit = self._to_str(row.get("Units"))
                elif py_name == "final_time" and not info.time_unit:
                    info.time_unit = self._to_str(row.get("Units"))
                continue

            unit = self._to_str(row.get("Units"))
            description = self._to_str(row.get("Comment"))

            definition = source_map.get(py_name)

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

        # Post-process stocks to identify inflows and outflows
        py_to_real = {}
        real_to_py = {}
        for _, row in doc.iterrows():
            rn = str(row.get("Real Name", "")).strip()
            pn = str(row.get("Py Name", "")).strip()
            if rn and pn:
                py_to_real[pn] = rn
                real_to_py[rn] = pn

        flow_real_names = {py_to_real.get(pn, pn) for pn in flow_py_names}

        for stock_var in info.stocks:
            py_name = real_to_py.get(stock_var.name, "")
            integ_name = f"_integ_{py_name}"

            eq_str = source_map.get(integ_name, "")
            if not eq_str:
                eq_str = info.raw_equations.get(stock_var.name, stock_var.equation)

            inflows, outflows = self._parse_stock_flows(
                eq_str, flow_real_names, py_to_real
            )
            stock_var.inflows = inflows
            stock_var.outflows = outflows

        return info, model

    @staticmethod
    def _to_str(value: Any) -> str:
        if value is None:
            return ""
        value_str = str(value).strip()
        return "" if value_str.lower() == "nan" else value_str

    @staticmethod
    def _extract_source_map(component_module: Any) -> dict[str, str]:
        """
        Extract all function definitions from the component module source in one go.
        Returns a mapping from py_name to the return expression.
        """
        import ast

        source_map = {}
        try:
            # Get source of the components module
            source = inspect.getsource(component_module)
            tree = ast.parse(source)

            # Walk the AST to extract function return expressions and stock Integ definitions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    for body_node in node.body:
                        if isinstance(body_node, ast.Return):
                            if body_node.value is not None:
                                source_map[node.name] = ast.unparse(
                                    body_node.value
                                ).strip()
                            break
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id.startswith(
                            "_integ_"
                        ):
                            source_map[target.id] = ast.unparse(node.value).strip()
        except Exception:
            pass
        return source_map

    @staticmethod
    def _detect_flow_py_names(source_map: dict[str, str]) -> set[str]:
        """
        Detect flow variable Python names by analyzing INTEG derivatives in the source map.

        Args:
            source_map: Map of variable names to their source expressions.

        Returns:
            Set of Python names that are likely flows.
        """
        flow_names: set[str] = set()
        # Look for variables called as functions inside Integ calls
        pattern = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")

        for name, expr in source_map.items():
            if not name.startswith("_integ_"):
                continue

            for match in pattern.findall(expr):
                if match not in {"lambda", "Integ"} and not match.startswith("_integ_"):
                    flow_names.add(match)

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
        if py_name in flow_py_names:
            return "flow"
        if comp_type_lower == "stateful":
            return "stock"
        if comp_type_lower == "constant":
            return "parameter"
        return "auxiliary"

    @staticmethod
    def _parse_stock_flows(
        equation_str: str,
        flow_real_names: set[str],
        py_to_real: dict[str, str],
    ) -> tuple[list[str], list[str]]:
        """
        Parse a stock's equation to identify inflows and outflows.
        """
        if not equation_str:
            return [], []

        inflows: list[str] = []
        outflows: list[str] = []

        tokens = re.split(r"(\s*[+\-]\s*)", equation_str)

        sign = "+"
        for token in tokens:
            stripped = token.strip()
            if stripped in ("+", "-"):
                sign = stripped
                continue
            if not stripped:
                continue

            func_calls = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\(\)", stripped)
            bare_names = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", stripped)

            for name in func_calls + bare_names:
                real_name = py_to_real.get(name, name)
                if real_name in flow_real_names:
                    if sign == "-":
                        outflows.append(real_name)
                    else:
                        inflows.append(real_name)

        # Deduplicate while preserving order
        inflows = list(dict.fromkeys(inflows))
        outflows = list(dict.fromkeys(outflows))

        return inflows, outflows
