from typing import Dict, List, Optional

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


class ModelSchema(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "file_name": "model.mdl",
                    "uploaded_at": "2026-03-24T12:00:00+00:00",
                    "parsed_with": "pysd",
                    "format": "mdl",
                    "stocks": [],
                    "flows": [],
                    "parameters": [],
                    "auxiliaries": [],
                }
            ]
        }
    )

    file_name: str = ""
    uploaded_at: str = ""
    parsed_with: str = ""
    format: str
    stocks: List[ModelVariableSchema] = Field(default_factory=list)
    flows: List[ModelVariableSchema] = Field(default_factory=list)
    parameters: List[ModelVariableSchema] = Field(default_factory=list)
    auxiliaries: List[ModelVariableSchema] = Field(default_factory=list)
    raw_equations: Dict[str, str] = Field(default_factory=dict, exclude=True)

    @property
    def all_variables(self) -> list[ModelVariableSchema]:
        return self.stocks + self.flows + self.parameters + self.auxiliaries

    def summary(self) -> str:
        return (
            f"File: {self.file_name} | Format: {self.format} | "
            f"Stocks: {len(self.stocks)} | Flows: {len(self.flows)} | "
            f"Parameters: {len(self.parameters)} | Auxiliaries: {len(self.auxiliaries)}"
        )

    def to_dict(self) -> dict:
        """
        Serialize model to dictionary for external consumption.
        """
        return {
            "file_name": self.file_name,
            "uploaded_at": self.uploaded_at,
            "parsed_with": self.parsed_with,
            "format": self.format,
            "stocks": [v.model_dump() for v in self.stocks],
            "flows": [v.model_dump() for v in self.flows],
            "parameters": [v.model_dump() for v in self.parameters],
            "auxiliaries": [v.model_dump() for v in self.auxiliaries],
        }
