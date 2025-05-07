"""
Utility module for maintainers to generate dataset_infos.json from Pydantic models.
"""

import datetime
import json
import types
from typing import Union, get_args, get_origin

import pyarrow as pa
from datasets import Features
from pydantic import BaseModel

from .models import EvalResult


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
    Build a Hugging Face Features object from a Pydantic BaseModel using PyArrow schema.
    """
    pa_fields = _schema_from_pydantic(model)
    pa_schema = pa.schema(pa_fields)
    return Features.from_arrow_schema(pa_schema)


def generate_dataset_infos(output_path: str = "dataset_infos.json"):
    """
    Generate a dataset_infos.json file from the EvalResult schema.
    """
    features = features_from_pydantic(EvalResult)
    infos = {"default": {"features": features.to_dict()}}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(infos, f, indent=2)
    print(f"Generated dataset_infos.json at {output_path}")
