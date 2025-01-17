from theflow.api_annotations import DeveloperAPI
from theflow.constants import DATE, MODEL_ECD
from theflow.schema import utils as schema_utils
from theflow.schema.encoders.base import BaseEncoderConfig
from theflow.schema.encoders.utils import EncoderDataclassField
from theflow.schema.features.base import BaseInputFeatureConfig
from theflow.schema.features.preprocessing.base import BasePreprocessingConfig
from theflow.schema.features.preprocessing.utils import PreprocessingDataclassField
from theflow.schema.features.utils import ecd_defaults_config_registry, ecd_input_config_registry, input_mixin_registry
from theflow.schema.utils import BaseMarshmallowConfig, theflow_dataclass


@DeveloperAPI
@ecd_defaults_config_registry.register(DATE)
@input_mixin_registry.register(DATE)
@theflow_dataclass
class DateInputFeatureConfigMixin(BaseMarshmallowConfig):
    """DateInputFeatureConfigMixin is a dataclass that configures the parameters used in both the date input
    feature and the date global defaults section of the The Flow Config."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=DATE)

    encoder: BaseEncoderConfig = EncoderDataclassField(
        MODEL_ECD,
        feature_type=DATE,
        default="embed",
    )


@DeveloperAPI
@ecd_input_config_registry.register(DATE)
@theflow_dataclass
class DateInputFeatureConfig(DateInputFeatureConfigMixin, BaseInputFeatureConfig):
    """DateInputFeature is a dataclass that configures the parameters used for a date input feature."""

    type: str = schema_utils.ProtectedString(DATE)
