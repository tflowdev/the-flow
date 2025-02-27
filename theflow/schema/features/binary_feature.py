from theflow.api_annotations import DeveloperAPI
from theflow.constants import BINARY, BINARY_WEIGHTED_CROSS_ENTROPY, MODEL_ECD, MODEL_GBM, ROC_AUC
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
@input_mixin_registry.register(BINARY)
@theflow_dataclass
class BinaryInputFeatureConfigMixin(BaseMarshmallowConfig):
    """BinaryInputFeatureConfigMixin is a dataclass that configures the parameters used in both the binary input
    feature and the binary global defaults section of the The Flow Config."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=BINARY)


@DeveloperAPI
@theflow_dataclass
class BinaryInputFeatureConfig(BinaryInputFeatureConfigMixin, BaseInputFeatureConfig):
    """BinaryInputFeatureConfig is a dataclass that configures the parameters used for a binary input feature."""

    type: str = schema_utils.ProtectedString(BINARY)

    encoder: BaseEncoderConfig = None


@DeveloperAPI
@ecd_input_config_registry.register(BINARY)
@theflow_dataclass
class ECDBinaryInputFeatureConfig(BinaryInputFeatureConfig):
    encoder: BaseEncoderConfig = EncoderDataclassField(
        MODEL_ECD,
        feature_type=BINARY,
        default="passthrough",
    )


@DeveloperAPI
@gbm_input_config_registry.register(BINARY)
@theflow_dataclass
class GBMBinaryInputFeatureConfig(BinaryInputFeatureConfig):
    encoder: BaseEncoderConfig = EncoderDataclassField(
        MODEL_GBM,
        feature_type=BINARY,
        default="passthrough",
    )


@DeveloperAPI
@gbm_defaults_config_registry.register(BINARY)
@theflow_dataclass
class GBMBinaryDefaultsConfig(BinaryInputFeatureConfigMixin):
    encoder: BaseEncoderConfig = EncoderDataclassField(
        MODEL_GBM,
        feature_type=BINARY,
        default="passthrough",
    )


@DeveloperAPI
@output_mixin_registry.register(BINARY)
@theflow_dataclass
class BinaryOutputFeatureConfigMixin(BaseMarshmallowConfig):
    """BinaryOutputFeatureConfigMixin is a dataclass that configures the parameters used in both the binary output
    feature and the binary global defaults section of the The Flow Config."""

    decoder: BaseDecoderConfig = None

    loss: BaseLossConfig = LossDataclassField(
        feature_type=BINARY,
        default=BINARY_WEIGHTED_CROSS_ENTROPY,
    )


@DeveloperAPI
@theflow_dataclass
class BinaryOutputFeatureConfig(BinaryOutputFeatureConfigMixin, BaseOutputFeatureConfig):
    """BinaryOutputFeatureConfig is a dataclass that configures the parameters used for a binary output feature."""

    type: str = schema_utils.ProtectedString(BINARY)

    calibration: bool = schema_utils.Boolean(
        default=False,
        description="Calibrate the model's output probabilities using temperature scaling.",
        parameter_metadata=FEATURE_METADATA[BINARY]["calibration"],
    )

    default_validation_metric: str = schema_utils.StringOptions(
        [ROC_AUC],
        default=ROC_AUC,
        description="Internal only use parameter: default validation metric for binary output feature.",
        parameter_metadata=INTERNAL_ONLY,
    )

    dependencies: list = schema_utils.List(
        default=[],
        description="List of input features that this feature depends on.",
        parameter_metadata=FEATURE_METADATA[BINARY]["dependencies"],
    )

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type="binary_output")

    reduce_dependencies: str = schema_utils.ReductionOptions(
        default="sum",
        description="How to reduce the dependencies of the output feature.",
        parameter_metadata=FEATURE_METADATA[BINARY]["reduce_dependencies"],
    )

    reduce_input: str = schema_utils.ReductionOptions(
        default="sum",
        description="How to reduce an input that is not a vector, but a matrix or a higher order tensor, on the first "
        "dimension (second if you count the batch dimension)",
        parameter_metadata=FEATURE_METADATA[BINARY]["reduce_input"],
    )

    threshold: float = schema_utils.FloatRange(
        default=0.5,
        min=0,
        max=1,
        description="The threshold used to convert output probabilities to predictions. Predicted probabilities greater"
        "than or equal to threshold are mapped to True.",
        parameter_metadata=FEATURE_METADATA[BINARY]["threshold"],
    )


@DeveloperAPI
@ecd_output_config_registry.register(BINARY)
@theflow_dataclass
class ECDBinaryOutputFeatureConfig(BinaryOutputFeatureConfig):
    decoder: BaseDecoderConfig = DecoderDataclassField(
        MODEL_ECD,
        feature_type=BINARY,
        default="regressor",
    )


@DeveloperAPI
@gbm_output_config_registry.register(BINARY)
@theflow_dataclass
class GBMBinaryOutputFeatureConfig(BinaryOutputFeatureConfig):
    decoder: BaseDecoderConfig = DecoderDataclassField(
        MODEL_GBM,
        feature_type=BINARY,
        default="regressor",
    )


@DeveloperAPI
@ecd_defaults_config_registry.register(BINARY)
@theflow_dataclass
class BinaryDefaultsConfig(BinaryInputFeatureConfigMixin, BinaryOutputFeatureConfigMixin):
    # NOTE(travis): defaults use ECD input feature as it contains all the encoders
    encoder: BaseEncoderConfig = EncoderDataclassField(
        MODEL_ECD,
        feature_type=BINARY,
        default="passthrough",
    )

    decoder: BaseDecoderConfig = DecoderDataclassField(
        MODEL_ECD,
        feature_type=BINARY,
        default="regressor",
    )
