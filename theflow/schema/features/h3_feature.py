from theflow.api_annotations import DeveloperAPI
from theflow.constants import H3, MODEL_ECD
from theflow.schema import utils as schema_utils
from theflow.schema.encoders.base import BaseEncoderConfig
from theflow.schema.encoders.utils import EncoderDataclassField
from theflow.schema.features.base import BaseInputFeatureConfig
from theflow.schema.features.preprocessing.base import BasePreprocessingConfig
from theflow.schema.features.preprocessing.utils import PreprocessingDataclassField
from theflow.schema.features.utils import ecd_defaults_config_registry, ecd_input_config_registry, input_mixin_registry
from theflow.schema.utils import BaseMarshmallowConfig, theflow_dataclass


@DeveloperAPI
@ecd_defaults_config_registry.register(H3)
@input_mixin_registry.register(H3)
@theflow_dataclass
class H3InputFeatureConfigMixin(BaseMarshmallowConfig):
    """H3InputFeatureConfigMixin is a dataclass that configures the parameters used in both the h3 input feature
    and the h3 global defaults section of the The Flow Config."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=H3)

    encoder: BaseEncoderConfig = EncoderDataclassField(
        MODEL_ECD,
        feature_type=H3,
        default="embed",
    )


@DeveloperAPI
@ecd_input_config_registry.register(H3)
@theflow_dataclass
class H3InputFeatureConfig(H3InputFeatureConfigMixin, BaseInputFeatureConfig):
    """H3InputFeatureConfig is a dataclass that configures the parameters used for an h3 input feature."""

    type: str = schema_utils.ProtectedString(H3)
