from theflow.api_annotations import DeveloperAPI
from theflow.constants import AUDIO, MODEL_ECD
from theflow.schema import utils as schema_utils
from theflow.schema.encoders.base import BaseEncoderConfig
from theflow.schema.encoders.utils import EncoderDataclassField
from theflow.schema.features.base import BaseInputFeatureConfig
from theflow.schema.features.preprocessing.base import BasePreprocessingConfig
from theflow.schema.features.preprocessing.utils import PreprocessingDataclassField
from theflow.schema.features.utils import ecd_defaults_config_registry, ecd_input_config_registry, input_mixin_registry
from theflow.schema.utils import BaseMarshmallowConfig, theflow_dataclass


@DeveloperAPI
@ecd_defaults_config_registry.register(AUDIO)
@input_mixin_registry.register(AUDIO)
@theflow_dataclass
class AudioInputFeatureConfigMixin(BaseMarshmallowConfig):
    """AudioInputFeatureConfigMixin is a dataclass that configures the parameters used in both the audio input
    feature and the audio global defaults section of the The Flow Config."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=AUDIO)

    encoder: BaseEncoderConfig = EncoderDataclassField(
        MODEL_ECD,
        feature_type=AUDIO,
        default="parallel_cnn",
    )


@DeveloperAPI
@ecd_input_config_registry.register(AUDIO)
@theflow_dataclass
class AudioInputFeatureConfig(AudioInputFeatureConfigMixin, BaseInputFeatureConfig):
    """AudioInputFeatureConfig is a dataclass that configures the parameters used for an audio input feature."""

    type: str = schema_utils.ProtectedString(AUDIO)
