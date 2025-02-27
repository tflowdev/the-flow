#! /usr/bin/env python
from typing import Optional

import pytest
from marshmallow.exceptions import ValidationError as MarshmallowValidationError
from marshmallow_dataclass import dataclass

import theflow.schema.optimizers as lso
from theflow.schema import utils as schema_utils

# Tests for custom dataclass/marshmallow fields:


def get_marshmallow_from_dataclass_field(dfield):
    """Helper method for checking marshmallow metadata succinctly."""
    return dfield.metadata["marshmallow_field"]


def test_torch_description_pull():
    example_empty_desc_prop = schema_utils.unload_jsonschema_from_marshmallow_class(lso.AdamOptimizerConfig)[
        "properties"
    ]["eps"]
    assert (
        isinstance(example_empty_desc_prop, dict)
        and "description" in example_empty_desc_prop
        and isinstance(example_empty_desc_prop["description"], str)
        and len(example_empty_desc_prop["description"]) > 3
    )


def test_OptimizerDataclassField():
    # Test default case:
    default_optimizer_field = lso.OptimizerDataclassField()
    assert default_optimizer_field.default_factory is not None
    assert get_marshmallow_from_dataclass_field(default_optimizer_field).allow_none is False
    assert default_optimizer_field.default_factory() == lso.AdamOptimizerConfig()

    # Test normal cases:
    optimizer_field = lso.OptimizerDataclassField("adamax")
    assert optimizer_field.default_factory is not None
    assert get_marshmallow_from_dataclass_field(optimizer_field).allow_none is False
    assert optimizer_field.default_factory() == lso.AdamaxOptimizerConfig()

    # Test invalid default case:
    with pytest.raises(AttributeError):
        lso.OptimizerDataclassField({})
    with pytest.raises(KeyError):
        lso.OptimizerDataclassField("test")
    with pytest.raises(AttributeError):
        lso.OptimizerDataclassField(1)

    # Test creating a schema with default options:
    @dataclass
    class CustomTestSchema(schema_utils.BaseMarshmallowConfig):
        foo: Optional[lso.BaseOptimizerConfig] = lso.OptimizerDataclassField()

    with pytest.raises(MarshmallowValidationError):
        CustomTestSchema.Schema().load({"foo": "test"})

    assert CustomTestSchema.Schema().load({}).foo == lso.AdamOptimizerConfig()

    # Test creating a schema with set default:
    @dataclass
    class CustomTestSchema(schema_utils.BaseMarshmallowConfig):
        foo: Optional[lso.BaseOptimizerConfig] = lso.OptimizerDataclassField("adamax")

    with pytest.raises(MarshmallowValidationError):
        CustomTestSchema.Schema().load({"foo": None})
    with pytest.raises(MarshmallowValidationError):
        CustomTestSchema.Schema().load({"foo": "test"})
    with pytest.raises(MarshmallowValidationError):
        CustomTestSchema.Schema().load({"foo": {"type": "invalid", "betas": (0.2, 0.2)}})

    assert CustomTestSchema.Schema().load(
        {"foo": {"type": "adamax", "betas": (0.2, 0.2)}}
    ).foo == lso.AdamaxOptimizerConfig(betas=(0.2, 0.2))
    assert CustomTestSchema.Schema().load(
        {"foo": {"type": "adamax", "betas": (0.2, 0.2), "extra_key": 1}}
    ).foo == lso.AdamaxOptimizerConfig(betas=(0.2, 0.2))


def test_ClipperDataclassField():
    # Test default case:
    default_clipper_field = lso.GradientClippingDataclassField(description="", default={})
    assert default_clipper_field.default_factory is not None
    assert get_marshmallow_from_dataclass_field(default_clipper_field).allow_none is True
    assert default_clipper_field.default_factory() == lso.GradientClippingConfig()

    # Test normal cases:
    clipper_field = lso.GradientClippingDataclassField(description="", default={"clipglobalnorm": 0.1})
    assert clipper_field.default_factory is not None
    assert get_marshmallow_from_dataclass_field(clipper_field).allow_none is True
    assert clipper_field.default_factory() == lso.GradientClippingConfig(clipglobalnorm=0.1)

    clipper_field = lso.GradientClippingDataclassField(description="", default={"clipglobalnorm": None})
    assert clipper_field.default_factory is not None
    assert get_marshmallow_from_dataclass_field(clipper_field).allow_none is True
    assert clipper_field.default_factory() == lso.GradientClippingConfig(clipglobalnorm=None)

    # Test invalid default case:
    with pytest.raises(MarshmallowValidationError):
        lso.GradientClippingDataclassField(description="", default="test")
    with pytest.raises(MarshmallowValidationError):
        lso.GradientClippingDataclassField(description="", default=None)
    with pytest.raises(MarshmallowValidationError):
        lso.GradientClippingDataclassField(description="", default=1)

    # Test creating a schema with set default:
    @dataclass
    class CustomTestSchema(schema_utils.BaseMarshmallowConfig):
        foo: Optional[lso.GradientClippingConfig] = lso.GradientClippingDataclassField(
            description="", default={"clipglobalnorm": 0.1}
        )

    with pytest.raises(MarshmallowValidationError):
        CustomTestSchema.Schema().load({"foo": "test"})
    with pytest.raises(MarshmallowValidationError):
        CustomTestSchema.Schema().load({"foo": {"clipglobalnorm": "invalid"}})

    assert CustomTestSchema.Schema().load({}).foo == lso.GradientClippingConfig(clipglobalnorm=0.1)
    assert CustomTestSchema.Schema().load({"foo": {"clipglobalnorm": 1}}).foo == lso.GradientClippingConfig(
        clipglobalnorm=1
    )
    assert CustomTestSchema.Schema().load(
        {"foo": {"clipglobalnorm": 1, "extra_key": 1}}
    ).foo == lso.GradientClippingConfig(clipglobalnorm=1)
