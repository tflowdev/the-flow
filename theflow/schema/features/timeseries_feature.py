from theflow.api_annotations import DeveloperAPI
from theflow.constants import HUBER, MEAN_SQUARED_ERROR, MODEL_ECD, TIMESERIES, VECTOR
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
@input_mixin_registry.register(TIMESERIES)
@theflow_dataclass
class TimeseriesInputFeatureConfigMixin(BaseMarshmallowConfig):
    """TimeseriesInputFeatureConfigMixin is a dataclass that configures the parameters used in both the timeseries
    input feature and the timeseries global defaults section of the The Flow Config."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=TIMESERIES)

    encoder: BaseEncoderConfig = EncoderDataclassField(
        MODEL_ECD,
        feature_type=TIMESERIES,
        default="parallel_cnn",
    )


@DeveloperAPI
@ecd_input_config_registry.register(TIMESERIES)
@theflow_dataclass
class TimeseriesInputFeatureConfig(TimeseriesInputFeatureConfigMixin, BaseInputFeatureConfig):
    """TimeseriesInputFeatureConfig is a dataclass that configures the parameters used for a timeseries input
    feature."""

    type: str = schema_utils.ProtectedString(TIMESERIES)


@DeveloperAPI
@output_mixin_registry.register(TIMESERIES)
@theflow_dataclass
class TimeseriesOutputFeatureConfigMixin(BaseMarshmallowConfig):
    """TimeseriesOutputFeatureConfigMixin configures the parameters used in both the timeseries output feature and
    the timeseries global defaults section of the The Flow Config."""

    decoder: BaseDecoderConfig = DecoderDataclassField(
        MODEL_ECD,
        feature_type=TIMESERIES,
        default="projector",
    )

    loss: BaseLossConfig = LossDataclassField(
        feature_type=TIMESERIES,
        default=HUBER,
    )


@DeveloperAPI
@ecd_output_config_registry.register(TIMESERIES)
@theflow_dataclass
class TimeseriesOutputFeatureConfig(BaseOutputFeatureConfig, TimeseriesOutputFeatureConfigMixin):
    """TimeseriesOutputFeatureConfig configures the parameters used for a timeseries output feature."""

    type: str = schema_utils.ProtectedString(TIMESERIES)

    dependencies: list = schema_utils.List(
        default=[],
        description="List of input features that this feature depends on.",
        parameter_metadata=FEATURE_METADATA[VECTOR]["dependencies"],
    )

    default_validation_metric: str = schema_utils.StringOptions(
        [MEAN_SQUARED_ERROR],
        default=MEAN_SQUARED_ERROR,
        description="Internal parameter.",
        parameter_metadata=INTERNAL_ONLY,
    )

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type="timeseries_output")

    reduce_dependencies: str = schema_utils.ReductionOptions(
        default=None,
        description="How to reduce the dependencies of the output feature.",
        parameter_metadata=FEATURE_METADATA[VECTOR]["reduce_dependencies"],
    )

    reduce_input: str = schema_utils.ReductionOptions(
        default=None,
        description="How to reduce an input that is not a vector, but a matrix or a higher order tensor, on the first "
        "dimension (second if you count the batch dimension)",
        parameter_metadata=FEATURE_METADATA[VECTOR]["reduce_input"],
    )

    horizon: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="Internal parameter. Obtained from preprocessing",
        parameter_metadata=INTERNAL_ONLY,
    )


@DeveloperAPI
@ecd_defaults_config_registry.register(TIMESERIES)
@theflow_dataclass
class TimeseriesDefaultsConfig(TimeseriesInputFeatureConfigMixin, TimeseriesOutputFeatureConfigMixin):
    pass
