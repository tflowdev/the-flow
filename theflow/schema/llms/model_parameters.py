from typing import Optional

from theflow.api_annotations import DeveloperAPI
from theflow.error import ConfigValidationError
from theflow.schema import utils as schema_utils
from theflow.schema.utils import theflow_dataclass


@DeveloperAPI
@theflow_dataclass
class RoPEScalingConfig(schema_utils.BaseMarshmallowConfig):
    """Dynamic RoPE-scaling (rotary position embeddings) to extend the context length of LLM like LLaMA, GPT-NeoX,
    or Falcon.

    This parameter is a dictionary containing the scaling configuration
    for the RoPE embeddings. Currently supports three scaling strategies: linear and dynamic. Their
    scaling factor must be an float greater than 1. The expected format is {'type': strategy name,
    'factor': scaling factor}
    """

    def __post_init__(self):
        # Both parameters must be set, or none.
        if not self.type:
            raise ConfigValidationError(
                f"`rope_scaling`'s `type` field must be one of ['linear', 'dynamic'], got {self.type}"
            )

        if not self.factor:
            raise ConfigValidationError(
                f"When using `rope_scaling`, `factor` must be specified and be > 1. Got {self.factor}."
            )

    type: Optional[str] = schema_utils.StringOptions(
        options=["linear", "dynamic"],
        default=None,
        allow_none=True,
        description="Currently supports two strategies: linear and dynamic scaling.",
    )

    factor: Optional[float] = schema_utils.FloatRange(
        default=None,
        allow_none=True,
        min=1.0,
        min_inclusive=False,
        description="The scaling factor for RoPE embeddings.",
    )


@DeveloperAPI
class RoPEScalingConfigField(schema_utils.DictMarshmallowField):
    def __init__(self):
        super().__init__(RoPEScalingConfig, default_missing=True)

    def _jsonschema_type_mapping(self):
        return schema_utils.unload_jsonschema_from_marshmallow_class(RoPEScalingConfig, title="rope_scaling")


@DeveloperAPI
@theflow_dataclass
class ModelParametersConfig(schema_utils.BaseMarshmallowConfig):
    rope_scaling: RoPEScalingConfig = RoPEScalingConfigField().get_default_field()

    neftune_noise_alpha: Optional[int] = schema_utils.IntegerRange(
        default=0,
        min=0,
        allow_none=True,
        description="The alpha parameter for the embedding noise, which controls the amount of noise added to the "
        "embeddings. The higher the value, the more noise is added. This is based on the paper NEFTune: Noisy "
        "Embeddings Improve Instruction Finetuning. Paper: https://arxiv.org/pdf/2310.05914.pdf. Default: 0."
        "Suggested values: 5, 10",
    )

    def to_dict(self):
        config = {}
        if self.rope_scaling:
            config["rope_scaling"] = self.rope_scaling.to_dict()
        if self.neftune_noise_alpha:
            config["neftune_noise_alpha"] = self.neftune_noise_alpha
        return config


@DeveloperAPI
class ModelParametersConfigField(schema_utils.DictMarshmallowField):
    def __init__(self):
        super().__init__(ModelParametersConfig, default_missing=True)

    def _jsonschema_type_mapping(self):
        return {
            "oneOf": [
                {"type": "null", "title": "disabled", "description": "Skip configurable model parameters."},
                {
                    **schema_utils.unload_jsonschema_from_marshmallow_class(ModelParametersConfig),
                    "title": "enabled",
                    "description": "Set model parameters options.",
                },
            ],
            "title": "Model Parameters",
            "description": "Configurable model parameters for LLMs.",
        }
