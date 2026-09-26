from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class PaperManifest(BaseModel):
    paper_id: str
    title: str | None = None
    content_list_path: Path
    kg_nodes_path: Path
    kg_edges_path: Path
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("paper_id")
    @classmethod
    def paper_id_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("paper_id must not be blank")
        return value


class RunConfig(BaseModel):
    run_id: str
    output_dir: Path
    paper: PaperManifest

    @field_validator("run_id")
    @classmethod
    def run_id_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("run_id must not be blank")
        return value


def load_run_config(path: Path) -> RunConfig:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigLoadError(f"Config file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigLoadError(f"Config file is not valid YAML: {path}") from exc

    if not isinstance(payload, dict):
        raise ConfigLoadError(f"Config file must contain a YAML mapping: {path}")

    try:
        return RunConfig.model_validate(payload)
    except ValueError as exc:
        raise ConfigLoadError(f"Config file failed validation: {path}") from exc


class ConfigLoadError(RuntimeError):
    pass
