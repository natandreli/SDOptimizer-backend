from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ModelVariableSchema(BaseModel):
    name: str
    type: str
    equation: str = ""
    unit: str = ""
    initial_value: Optional[float] = None
    description: str = ""
    inflows: List[str] = Field(default_factory=list)
    outflows: List[str] = Field(default_factory=list)


class ModelInformationSchema(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "source_file": "model.mdl",
                    "format": "mdl",
                    "stocks": [],
                    "flows": [],
                    "parameters": [],
                    "auxiliaries": [],
                }
            ]
        }
    )

    source_file: str
    format: str
    stocks: List[ModelVariableSchema] = Field(default_factory=list)
    flows: List[ModelVariableSchema] = Field(default_factory=list)
    parameters: List[ModelVariableSchema] = Field(default_factory=list)
    auxiliaries: List[ModelVariableSchema] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    raw_equations: Dict[str, str] = Field(default_factory=dict, exclude=True)

    @property
    def all_variables(self) -> list[ModelVariableSchema]:
        return self.stocks + self.flows + self.parameters + self.auxiliaries

    def summary(self) -> str:
        return (
            f"File: {self.source_file} | Format: {self.format} | "
            f"Stocks: {len(self.stocks)} | Flows: {len(self.flows)} | "
            f"Parameters: {len(self.parameters)} | Auxiliaries: {len(self.auxiliaries)}"
        )

    def to_dict(self) -> dict:
        """
        Serialize model to dictionary for external consumption.
        """
        return {
            "source_file": self.source_file,
            "format": self.format,
            "stocks": [v.model_dump() for v in self.stocks],
            "flows": [v.model_dump() for v in self.flows],
            "parameters": [v.model_dump() for v in self.parameters],
            "auxiliaries": [v.model_dump() for v in self.auxiliaries],
        }
