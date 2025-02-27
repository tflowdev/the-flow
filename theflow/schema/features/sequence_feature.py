from theflow.api_annotations import DeveloperAPI
from theflow.constants import LOSS, MODEL_ECD, SEQUENCE, SEQUENCE_SOFTMAX_CROSS_ENTROPY
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
    input_mixin_registry,
    output_mixin_registry,
)
from theflow.schema.metadata import FEATURE_METADATA
from theflow.schema.metadata.parameter_metadata import INTERNAL_ONLY
from theflow.schema.utils import BaseMarshmallowConfig, theflow_dataclass


@DeveloperAPI
@input_mixin_registry.register(SEQUENCE)
@theflow_dataclass
class SequenceInputFeatureConfigMixin(BaseMarshmallowConfig):
    """SequenceInputFeatureConfigMixin is a dataclass that configures the parameters used in both the sequence
    input feature and the sequence global defaults section of the The Flow Config."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=SEQUENCE)

    encoder: BaseEncoderConfig = EncoderDataclassField(
        MODEL_ECD,
        feature_type=SEQUENCE,
        default="embed",
    )


@DeveloperAPI
@ecd_input_config_registry.register(SEQUENCE)
@theflow_dataclass
class SequenceInputFeatureConfig(SequenceInputFeatureConfigMixin, BaseInputFeatureConfig):
    """SequenceInputFeatureConfig is a dataclass that configures the parameters used for a sequence input
    feature."""

    type: str = schema_utils.ProtectedString(SEQUENCE)


@DeveloperAPI
@output_mixin_registry.register(SEQUENCE)
@theflow_dataclass
class SequenceOutputFeatureConfigMixin(BaseMarshmallowConfig):
    """SequenceOutputFeatureConfigMixin is a dataclass that configures the parameters used in both the sequence
    output feature and the sequence global defaults section of the The Flow Config."""

    decoder: BaseDecoderConfig = DecoderDataclassField(
        MODEL_ECD,
        feature_type=SEQUENCE,
        default="generator",
    )

    loss: BaseLossConfig = LossDataclassField(
        feature_type=SEQUENCE,
        default=SEQUENCE_SOFTMAX_CROSS_ENTROPY,
    )


@DeveloperAPI
@ecd_output_config_registry.register(SEQUENCE)
@theflow_dataclass
class SequenceOutputFeatureConfig(SequenceOutputFeatureConfigMixin, BaseOutputFeatureConfig):
    """SequenceOutputFeatureConfig is a dataclass that configures the parameters used for a sequence output
    feature."""

    type: str = schema_utils.ProtectedString(SEQUENCE)

    default_validation_metric: str = schema_utils.StringOptions(
        [LOSS],
        default=LOSS,
        description="Internal only use parameter: default validation metric for sequence output feature.",
        parameter_metadata=INTERNAL_ONLY,
    )

    dependencies: list = schema_utils.List(
        default=[],
        description="List of input features that this feature depends on.",
        parameter_metadata=FEATURE_METADATA[SEQUENCE]["dependencies"],
    )

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type="sequence_output")

    reduce_dependencies: str = schema_utils.ReductionOptions(
        default="sum",
        description="How to reduce the dependencies of the output feature.",
        parameter_metadata=FEATURE_METADATA[SEQUENCE]["reduce_dependencies"],
    )

    reduce_input: str = schema_utils.ReductionOptions(
        default="sum",
        description="How to reduce an input that is not a vector, but a matrix or a higher order tensor, on the first "
        "dimension (second if you count the batch dimension)",
        parameter_metadata=FEATURE_METADATA[SEQUENCE]["reduce_input"],
    )


@DeveloperAPI
@ecd_defaults_config_registry.register(SEQUENCE)
@theflow_dataclass
class SequenceDefaultsConfig(SequenceInputFeatureConfigMixin, SequenceOutputFeatureConfigMixin):
    pass
