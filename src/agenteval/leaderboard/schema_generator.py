"""
Utility module for maintainers to generate HuggingFace Dataset schema from Pydantic models.
"""

import datetime
import types
from importlib import resources
from typing import Union, get_args, get_origin

import pyarrow as pa
import yaml
from datasets import Features
from pydantic import BaseModel

from ..models import EvalResult


def _pa_type_for_annotation(anno) -> pa.DataType:
    origin = get_origin(anno)
    # Handle Optional and Union
    if origin is Union or origin is types.UnionType:
        args = [a for a in get_args(anno) if a is not type(None)]
        if len(args) == 1:
            return _pa_type_for_annotation(args[0])
        else:
            raise ValueError(f"Unsupported Union annotation {anno}")
    # Primitives
    if anno is str:
        return pa.string()
    if anno is int:
        return pa.int64()
    if anno is float:
        return pa.float64()
    if anno is bool:
        return pa.bool_()
    # Datetime
    if anno is datetime.datetime:
        return pa.timestamp("us")
    # Lists
    if origin is list:
        inner = get_args(anno)[0]
        return pa.list_(_pa_type_for_annotation(inner))
    # Nested BaseModel
    if isinstance(anno, type) and issubclass(anno, BaseModel):
        inner_schema = _schema_from_pydantic(anno)
        return pa.struct(inner_schema)
    raise ValueError(f"Unsupported annotation {anno}")


def _schema_from_pydantic(model: type[BaseModel]) -> list[pa.Field]:
    fields = []
    for name, field in model.model_fields.items():
        if getattr(field, "exclude", False):
            continue
        if name == "submit_time":
            pa_type = pa.timestamp("us", tz="UTC")
        else:
            pa_type = _pa_type_for_annotation(field.annotation)
        fields.append(pa.field(name, pa_type))
    return fields


def features_from_pydantic(model: type[BaseModel]) -> Features:
    """
    Build a HuggingFace Features object from a Pydantic BaseModel using PyArrow schema.
    """
    pa_fields = _schema_from_pydantic(model)
    pa_schema = pa.schema(pa_fields)
    return Features.from_arrow_schema(pa_schema)


def write_dataset_features(output_path: str) -> None:
    """
    Write the HuggingFace Features data inferred from the EvalResult schema.
    """
    features = features_from_pydantic(EvalResult)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml_values = features._to_yaml_list()
        yaml.safe_dump(yaml_values, f, indent=2, sort_keys=False)


def load_dataset_features(input_path: str | None = None) -> Features:
    """
    Load the HuggingFace Features data from a YAML file.
    """
    if input_path is None:
        # load the shipped dataset_features.yml from the package
        with resources.open_text(
            __package__, "dataset_features.yml", encoding="utf-8"
        ) as f:
            yaml_values = yaml.safe_load(f)
    else:
        with open(input_path, "r", encoding="utf-8") as f:
            yaml_values = yaml.safe_load(f)
    return Features._from_yaml_list(yaml_values)
