from theflow.api_annotations import DeveloperAPI
from theflow.constants import TEXT
from theflow.schema import utils as schema_utils
from theflow.schema.defaults.base import BaseDefaultsConfig
from theflow.schema.defaults.utils import DefaultsDataclassField
from theflow.schema.features.base import BaseFeatureConfig
from theflow.schema.features.utils import llm_defaults_config_registry
from theflow.schema.utils import theflow_dataclass


@DeveloperAPI
@theflow_dataclass
class LLMDefaultsConfig(BaseDefaultsConfig):
    text: BaseFeatureConfig = DefaultsDataclassField(feature_type=TEXT, defaults_registry=llm_defaults_config_registry)


@DeveloperAPI
class LLMDefaultsField(schema_utils.DictMarshmallowField):
    def __init__(self):
        super().__init__(LLMDefaultsConfig)
