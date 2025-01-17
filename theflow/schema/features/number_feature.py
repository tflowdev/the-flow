from typing import List, Tuple, Union

from theflow.api_annotations import DeveloperAPI
from theflow.constants import MEAN_SQUARED_ERROR, MODEL_ECD, MODEL_GBM, NUMBER
from theflow.schema import utils as schema_utils
from theflow.schema.decoders.base import BaseDecoderConfig
from theflow.schema.decoders.utils import DecoderDataclassField
from theflow.schema.encoders.base import BaseEncoderConfig
from theflow.schema.encoders.utils import EncoderDataclassField
from theflow.schema.features.base import BaseInputFeatureConfig, BaseOutputFeatureConfig
from theflow.schema.features.loss.loss import BaseLossConfig
from theflow.schema.features.loss.utils import LossDataclassField
from theflow.schema.features.preprocessing.base import BasePreprocessingConfig
from theflow.schema.features.preprocessing.utils import PreprocessingDataclassField
from theflow.schema.features.utils import (
    ecd_defaults_config_registry,
    ecd_input_config_registry,
    ecd_output_config_registry,
    gbm_defaults_config_registry,
    gbm_input_config_registry,
    gbm_output_config_registry,
    input_mixin_registry,
    output_mixin_registry,
)
from theflow.schema.metadata import FEATURE_METADATA
from theflow.schema.metadata.parameter_metadata import INTERNAL_ONLY
from theflow.schema.utils import BaseMarshmallowConfig, theflow_dataclass


@DeveloperAPI
@input_mixin_registry.register(NUMBER)
@theflow_dataclass
class NumberInputFeatureConfigMixin(BaseMarshmallowConfig):
    """NumberInputFeatureConfigMixin is a dataclass that configures the parameters used in both the number input
    feature and the number global defaults section of the The Flow Config."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=NUMBER)


@DeveloperAPI
@theflow_dataclass
class NumberInputFeatureConfig(NumberInputFeatureConfigMixin, BaseInputFeatureConfig):
    """NumberInputFeatureConfig is a dataclass that configures the parameters used for a number input feature."""

    type: str = schema_utils.ProtectedString(NUMBER)

    encoder: BaseEncoderConfig = None


@DeveloperAPI
@ecd_input_config_registry.register(NUMBER)
@theflow_dataclass
class ECDNumberInputFeatureConfig(NumberInputFeatureConfig):
    encoder: BaseEncoderConfig = EncoderDataclassField(
        MODEL_ECD,
        feature_type=NUMBER,
        default="passthrough",
    )


@DeveloperAPI
@gbm_input_config_registry.register(NUMBER)
@theflow_dataclass
class GBMNumberInputFeatureConfig(NumberInputFeatureConfig):
    encoder: BaseEncoderConfig = EncoderDataclassField(
        MODEL_GBM,
        feature_type=NUMBER,
        default="passthrough",
    )


@DeveloperAPI
@gbm_defaults_config_registry.register(NUMBER)
@theflow_dataclass
class GBMNumberDefaultsConfig(NumberInputFeatureConfigMixin):
    encoder: BaseEncoderConfig = EncoderDataclassField(
        MODEL_GBM,
        feature_type=NUMBER,
        default="passthrough",
    )


@DeveloperAPI
@output_mixin_registry.register(NUMBER)
@theflow_dataclass
class NumberOutputFeatureConfigMixin(BaseMarshmallowConfig):
    """NumberOutputFeatureConfigMixin is a dataclass that configures the parameters used in both the number output
    feature and the number global defaults section of the The Flow Config."""

    decoder: BaseDecoderConfig = None

    loss: BaseLossConfig = LossDataclassField(
        feature_type=NUMBER,
        default=MEAN_SQUARED_ERROR,
    )


@DeveloperAPI
@theflow_dataclass
class NumberOutputFeatureConfig(NumberOutputFeatureConfigMixin, BaseOutputFeatureConfig):
    """NumberOutputFeatureConfig is a dataclass that configures the parameters used for a category output
    feature."""

    type: str = schema_utils.ProtectedString(NUMBER)

    clip: Union[List[int], Tuple[int]] = schema_utils.FloatRangeTupleDataclassField(
        n=2,
        default=None,
        allow_none=True,
        min=0,
        max=999999999,
        description="Clip the predicted output to the specified range.",
        parameter_metadata=FEATURE_METADATA[NUMBER]["clip"],
    )

    default_validation_metric: str = schema_utils.StringOptions(
        [MEAN_SQUARED_ERROR],
        default=MEAN_SQUARED_ERROR,
        description="Internal only use parameter: default validation metric for number output feature.",
        parameter_metadata=INTERNAL_ONLY,
    )

    dependencies: list = schema_utils.List(
        default=[],
        description="List of input features that this feature depends on.",
        parameter_metadata=FEATURE_METADATA[NUMBER]["dependencies"],
    )

    reduce_dependencies: str = schema_utils.ReductionOptions(
        default="sum",
        description="How to reduce the dependencies of the output feature.",
        parameter_metadata=FEATURE_METADATA[NUMBER]["reduce_dependencies"],
    )

    reduce_input: str = schema_utils.ReductionOptions(
        default="sum",
        description="How to reduce an input that is not a vector, but a matrix or a higher order tensor, on the first "
        "dimension (second if you count the batch dimension)",
        parameter_metadata=FEATURE_METADATA[NUMBER]["reduce_input"],
    )

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type="number_output")


@DeveloperAPI
@ecd_output_config_registry.register(NUMBER)
@theflow_dataclass
class ECDNumberOutputFeatureConfig(NumberOutputFeatureConfig):
    decoder: BaseDecoderConfig = DecoderDataclassField(
        MODEL_ECD,
        feature_type=NUMBER,
        default="regressor",
    )


@DeveloperAPI
@gbm_output_config_registry.register(NUMBER)
@theflow_dataclass
class GBMNumberOutputFeatureConfig(NumberOutputFeatureConfig):
    decoder: BaseDecoderConfig = DecoderDataclassField(
        MODEL_GBM,
        feature_type=NUMBER,
        default="regressor",
    )


@DeveloperAPI
@ecd_defaults_config_registry.register(NUMBER)
@theflow_dataclass
class NumberDefaultsConfig(NumberInputFeatureConfigMixin, NumberOutputFeatureConfigMixin):
    encoder: BaseEncoderConfig = EncoderDataclassField(
        MODEL_ECD,
        feature_type=NUMBER,
        default="passthrough",
    )

    decoder: BaseDecoderConfig = DecoderDataclassField(
        MODEL_ECD,
        feature_type=NUMBER,
        default="regressor",
    )
