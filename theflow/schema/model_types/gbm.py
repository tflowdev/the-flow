from typing import Optional

from theflow.api_annotations import DeveloperAPI
from theflow.schema import utils as schema_utils
from theflow.schema.defaults.gbm import GBMDefaultsConfig, GBMDefaultsField
from theflow.schema.features.base import (
    BaseInputFeatureConfig,
    BaseOutputFeatureConfig,
    FeatureCollection,
    GBMInputFeatureSelection,
    GBMOutputFeatureSelection,
)
from theflow.schema.hyperopt import HyperoptConfig, HyperoptField
from theflow.schema.model_types.base import ModelConfig, register_model_type
from theflow.schema.preprocessing import PreprocessingConfig, PreprocessingField
from theflow.schema.trainer import GBMTrainerConfig, GBMTrainerField
from theflow.schema.utils import theflow_dataclass


@DeveloperAPI
@register_model_type(name="gbm")
@theflow_dataclass
class GBMModelConfig(ModelConfig):
    """Parameters for GBM."""

    model_type: str = schema_utils.ProtectedString("gbm")

    input_features: FeatureCollection[BaseInputFeatureConfig] = GBMInputFeatureSelection().get_list_field()
    output_features: FeatureCollection[BaseOutputFeatureConfig] = GBMOutputFeatureSelection().get_list_field()

    trainer: GBMTrainerConfig = GBMTrainerField().get_default_field()
    preprocessing: PreprocessingConfig = PreprocessingField().get_default_field()
    defaults: GBMDefaultsConfig = GBMDefaultsField().get_default_field()
    hyperopt: Optional[HyperoptConfig] = HyperoptField().get_default_field()
