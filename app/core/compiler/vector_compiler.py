from __future__ import annotations

import ast
import logging
from pathlib import Path

import numpy as np
import pysd

logger = logging.getLogger("vector_compiler")


class VectorCompilationError(Exception):
    """Raised when AST compilation or execution fails."""

    pass


class VectorModelCompiler:
    """
    Dynamic Vensim-to-NumPy Vectorised Compiler

    Translates PySD models into pure NumPy mathematical integration loops.
    """

    def __init__(self, model_py_path: str | Path, pysd_model: pysd.PySD) -> None:
        self.model_py_path = Path(model_py_path)
        self.model = pysd_model
        self.doc = pysd_model.doc

        self.stocks: list[str] = []
        self.constants: list[str] = []
        self.auxiliaries: list[str] = []

        self.stock_map: dict[str, int] = {}
        self.param_map: dict[str, int] = {}
        self.name_to_pyname: dict[str, str] = {}
        self.pyname_to_realname: dict[str, str] = {}

        self._simulate_fn = None
        self._derivative_fn = None
        self._evaluate_vars_fn = None
        self._stock_initial_fns: dict[str, callable] = {}

        # Initialize mappings
        self._initialize_mappings()

    def _initialize_mappings(self) -> None:
        """Extract stocks, constants, and auxiliaries from doc and establish mappings."""
        doc = self.doc

        self.stocks = doc[doc["Type"] == "Stateful"]["Py Name"].tolist()
        self.constants = doc[doc["Type"] == "Constant"]["Py Name"].tolist()
        self.auxiliaries = doc[doc["Type"] == "Auxiliary"]["Py Name"].tolist()

        control_vars = {"saveper", "time_step", "final_time", "initial_time", "time"}
        self.constants = [c for c in self.constants if c not in control_vars]
        self.auxiliaries = [a for a in self.auxiliaries if a not in control_vars]

        self.stock_map = {name: idx for idx, name in enumerate(self.stocks)}
        self.param_map = {name: idx for idx, name in enumerate(self.constants)}

        for _, row in doc.iterrows():
            real = row["Real Name"]
            py = row["Py Name"]
            if isinstance(real, str) and isinstance(py, str):
                self.name_to_pyname[real] = py
                self.name_to_pyname[real.strip()] = py
                self.pyname_to_realname[py] = real

    def compile(self) -> VectorModelCompiler:
        """Parses PySD Python file AST, restructures, topological sorts, and compiles equations."""
        try:
            with open(self.model_py_path, "r", encoding="utf-8") as f:
                source = f.read()
            tree = ast.parse(source)

            function_defs = {}
            integ_assigns = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    function_defs[node.name] = node
                elif isinstance(node, ast.Assign):
                    if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                        target_name = node.targets[0].id
                        if target_name.startswith("_integ_"):
                            integ_assigns[target_name] = node.value

            known_vars = set(self.stocks + self.constants + self.auxiliaries + ["time"])

            class PySDFunctionCallTransformer(ast.NodeTransformer):
                def visit_Call(self, node):
                    self.generic_visit(node)
                    if isinstance(node.func, ast.Name) and node.func.id in known_vars:
                        return ast.Name(id=node.func.id, ctx=ast.Load())
                    return node

            transformer = PySDFunctionCallTransformer()

            # Extract auxiliary and constant equations
            aux_equations = {}
            for name in self.auxiliaries + self.constants:
                node = function_defs.get(name)
                if node:
                    for stmt in node.body:
                        if isinstance(stmt, ast.Return):
                            cleaned_expr = transformer.visit(stmt.value)
                            aux_equations[name] = ast.unparse(cleaned_expr)

            # Extract stock flow/derivatives and initial expressions
            stock_equations = {}
            stock_initials = {}
            for stock_name in self.stocks:
                integ_name = f"_integ_{stock_name}"
                integ_call = integ_assigns.get(integ_name)
                if integ_call and isinstance(integ_call, ast.Call):
                    flow_arg = integ_call.args[0]
                    if isinstance(flow_arg, ast.Lambda):
                        cleaned_flow = transformer.visit(flow_arg.body)
                        stock_equations[stock_name] = ast.unparse(cleaned_flow)
                    else:
                        cleaned_flow = transformer.visit(flow_arg)
                        stock_equations[stock_name] = ast.unparse(cleaned_flow)

                    init_arg = integ_call.args[1]
                    if isinstance(init_arg, ast.Lambda):
                        cleaned_init = transformer.visit(init_arg.body)
                        stock_initials[stock_name] = ast.unparse(cleaned_init)
                    else:
                        cleaned_init = transformer.visit(init_arg)
                        stock_initials[stock_name] = ast.unparse(cleaned_init)

            # Topological sort of auxiliary dependencies
            dependencies = {}
            for name in self.auxiliaries:
                node = function_defs.get(name)
                if node:
                    deps = set()
                    for subnode in ast.walk(node):
                        if isinstance(subnode, ast.Call) and isinstance(
                            subnode.func, ast.Name
                        ):
                            func_id = subnode.func.id
                            if func_id in self.auxiliaries:
                                deps.add(func_id)
                    dependencies[name] = deps

            def topological_sort(nodes, deps_dict):
                visited = set()
                temp_visited = set()
                order = []

                def visit(node):
                    if node in temp_visited:
                        raise ValueError(f"Cyclic dependency detected at: {node}")
                    if node not in visited:
                        temp_visited.add(node)
                        for dep in deps_dict.get(node, []):
                            if dep in nodes:
                                visit(dep)
                        temp_visited.remove(node)
                        visited.add(node)
                        order.append(node)

                for node in nodes:
                    if node not in visited:
                        visit(node)
                return order

            sorted_aux = topological_sort(self.auxiliaries, dependencies)

            # Generate Pure Python/NumPy code string
            code_lines = ["import numpy as np", ""]
            code_lines.append("def derivative_fn(S, params, time):")

            for stock_name, idx in self.stock_map.items():
                code_lines.append(f"    {stock_name} = S[{idx}]")

            for param_name, idx in self.param_map.items():
                code_lines.append(f"    {param_name} = params[{idx}]")

            for name in sorted_aux:
                expr = aux_equations.get(name)
                if expr is not None:
                    code_lines.append(f"    {name} = {expr}")
                else:
                    code_lines.append(f"    {name} = 0.0")

            deriv_vars = []
            for stock_name in self.stocks:
                expr = stock_equations.get(stock_name)
                var_name = f"d_{stock_name}"
                if expr is not None:
                    code_lines.append(f"    {var_name} = {expr}")
                else:
                    code_lines.append(f"    {var_name} = 0.0")
                deriv_vars.append(var_name)

            code_lines.append(
                f"    return np.array([{', '.join(deriv_vars)}], dtype=np.float64)"
            )
            code_lines.append("")

            # Main integration function
            code_lines.extend(
                [
                    "def simulate_fn(initial_state, params, dt, total_steps, record_interval):",
                    "    num_recorded = int(total_steps / record_interval) + 1",
                    "    trajectory = np.zeros((num_recorded, len(initial_state)), dtype=np.float64)",
                    "    times = np.zeros(num_recorded, dtype=np.float64)",
                    "",
                    "    S = initial_state.copy()",
                    "    trajectory[0] = S",
                    "    times[0] = 0.0",
                    "",
                    "    rec_idx = 1",
                    "    for step in range(1, total_steps + 1):",
                    "        time_val = step * dt",
                    "        dS = derivative_fn(S, params, time_val)",
                    "        S = S + dS * dt",
                    "",
                    "        if step % record_interval == 0:",
                    "            if rec_idx < num_recorded:",
                    "                trajectory[rec_idx] = S",
                    "                times[rec_idx] = time_val",
                    "                rec_idx += 1",
                    "",
                    "    return times, trajectory",
                ]
            )

            # Trajectory Evaluator for auxiliary downsampling
            eval_lines = ["def evaluate_vars_fn(trajectory, params, times):"]
            eval_lines.append("    n_steps = len(times)")

            for col_py in self.constants + self.auxiliaries:
                eval_lines.append(
                    f"    arr_{col_py} = np.zeros(n_steps, dtype=np.float64)"
                )

            eval_lines.append("    for i in range(n_steps):")
            eval_lines.append("        time = times[i]")
            for stock_name, idx in self.stock_map.items():
                eval_lines.append(f"        {stock_name} = trajectory[i, {idx}]")
            for param_name, idx in self.param_map.items():
                eval_lines.append(f"        {param_name} = params[{idx}]")

            for name in sorted_aux:
                expr = aux_equations.get(name)
                if expr is not None:
                    eval_lines.append(f"        {name} = {expr}")
                else:
                    eval_lines.append(f"        {name} = 0.0")

            for col_py in self.constants + self.auxiliaries:
                eval_lines.append(f"        arr_{col_py}[i] = {col_py}")

            ret_vars = [f"arr_{col_py}" for col_py in self.constants + self.auxiliaries]
            eval_lines.append(f"    return ({', '.join(ret_vars)})")

            # Execute derivative and simulation functions
            namespace = {}
            exec("\n".join(code_lines), namespace)
            self._simulate_fn = namespace["simulate_fn"]
            self._derivative_fn = namespace["derivative_fn"]

            # Execute evaluator function
            eval_ns = {"np": np}
            exec("\n".join(eval_lines), eval_ns)
            self._evaluate_vars_fn = eval_ns["evaluate_vars_fn"]

            # Stock initials functions
            self._stock_initial_fns = {}
            for stock_name, expr in stock_initials.items():
                func_code = "def init_fn(params):\n"
                for param_name, idx in self.param_map.items():
                    func_code += f"    {param_name} = params[{idx}]\n"
                func_code += f"    return {expr}"
                init_ns = {"np": np}
                exec(func_code, init_ns)
                self._stock_initial_fns[stock_name] = init_ns["init_fn"]

            logger.info(
                "Successfully compiled model using Pure Python Vectorised Compiler."
            )
            return self

        except Exception as e:
            raise VectorCompilationError(f"AST Compilation failed: {e}") from e

    def simulate(
        self,
        parameter_overrides: dict[str, float] | None = None,
        dt: float = 0.25,
        total_time: float = 100.0,
        return_timestamps: np.ndarray | list[float] | None = None,
        return_columns: list[str] | None = None,
    ) -> dict[str, list[float]]:
        """Executes simulation with extreme NumPy vectorized speed."""
        if self._simulate_fn is None:
            raise VectorCompilationError(
                "Compiler must be compiled before simulate is called."
            )

        overrides = {}
        if parameter_overrides:
            for k, v in parameter_overrides.items():
                py_name = self.name_to_pyname.get(k, k)
                overrides[py_name] = v

        # Parameter array mapping
        params_arr = np.zeros(len(self.constants), dtype=np.float64)
        for name, idx in self.param_map.items():
            if name in overrides:
                params_arr[idx] = float(overrides[name])
            else:
                try:
                    val_fn = getattr(self.model.components, name)
                    params_arr[idx] = float(val_fn())
                except Exception:
                    row = self.doc[self.doc["Py Name"] == name]
                    if not row.empty:
                        params_arr[idx] = float(row.iloc[0]["Initial Value"] or 0.0)
                    else:
                        params_arr[idx] = 0.0

        # Initial state mapping
        initial_state_arr = np.zeros(len(self.stocks), dtype=np.float64)
        for name, idx in self.stock_map.items():
            if name in overrides:
                initial_state_arr[idx] = float(overrides[name])
            else:
                try:
                    initial_state_arr[idx] = float(
                        self._stock_initial_fns[name](params_arr)
                    )
                except Exception:
                    row = self.doc[self.doc["Py Name"] == name]
                    if not row.empty:
                        initial_state_arr[idx] = float(
                            row.iloc[0]["Initial Value"] or 0.0
                        )
                    else:
                        initial_state_arr[idx] = 0.0

        total_steps = int(total_time / dt)
        if total_steps <= 0:
            total_steps = 1

        # Handle record interval mapping
        if return_timestamps is not None:
            return_timestamps = np.asarray(return_timestamps)
            if len(return_timestamps) > 1:
                avg_diff = np.mean(np.diff(return_timestamps))
                record_interval = max(1, int(round(avg_diff / dt)))
            else:
                record_interval = total_steps
        else:
            record_interval = 1

        # Run integration loop
        times, trajectory = self._simulate_fn(
            initial_state_arr, params_arr, dt, total_steps, record_interval
        )

        # Assemble output series
        result_series = {"time": times.tolist()}

        # Populate stocks
        for stock_name, idx in self.stock_map.items():
            real_name = self.pyname_to_realname.get(stock_name, stock_name)
            if (
                return_columns is None
                or real_name in return_columns
                or stock_name in return_columns
            ):
                result_series[real_name] = trajectory[:, idx].tolist()

        # Evaluate and populate constants and auxiliaries
        evaluated_arrays = self._evaluate_vars_fn(trajectory, params_arr, times)
        all_other_vars = self.constants + self.auxiliaries
        for idx_var, col_py in enumerate(all_other_vars):
            real_name = self.pyname_to_realname.get(col_py, col_py)
            if (
                return_columns is None
                or real_name in return_columns
                or col_py in return_columns
            ):
                result_series[real_name] = evaluated_arrays[idx_var].tolist()

        return result_series
