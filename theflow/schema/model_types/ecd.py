from typing import Optional

from theflow.api_annotations import DeveloperAPI
from theflow.schema import utils as schema_utils
from theflow.schema.combiners.base import BaseCombinerConfig
from theflow.schema.combiners.utils import CombinerSelection
from theflow.schema.defaults.ecd import ECDDefaultsConfig, ECDDefaultsField
from theflow.schema.features.base import (
    BaseInputFeatureConfig,
    BaseOutputFeatureConfig,
    ECDInputFeatureSelection,
    ECDOutputFeatureSelection,
    FeatureCollection,
)
from theflow.schema.hyperopt import HyperoptConfig, HyperoptField
from theflow.schema.model_types.base import ModelConfig, register_model_type
from theflow.schema.preprocessing import PreprocessingConfig, PreprocessingField
from theflow.schema.trainer import ECDTrainerConfig, ECDTrainerField
from theflow.schema.utils import theflow_dataclass


@DeveloperAPI
@register_model_type(name="ecd")
@theflow_dataclass
class ECDModelConfig(ModelConfig):
    """Parameters for ECD."""

    model_type: str = schema_utils.ProtectedString("ecd")

    input_features: FeatureCollection[BaseInputFeatureConfig] = ECDInputFeatureSelection().get_list_field()
    output_features: FeatureCollection[BaseOutputFeatureConfig] = ECDOutputFeatureSelection().get_list_field()

    combiner: BaseCombinerConfig = CombinerSelection().get_default_field()

    trainer: ECDTrainerConfig = ECDTrainerField().get_default_field()
    preprocessing: PreprocessingConfig = PreprocessingField().get_default_field()
    defaults: ECDDefaultsConfig = ECDDefaultsField().get_default_field()
    hyperopt: Optional[HyperoptConfig] = HyperoptField().get_default_field()
