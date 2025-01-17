from theflow.api_annotations import DeveloperAPI
from theflow.constants import BINARY, CATEGORY, NUMBER
from theflow.schema import utils as schema_utils
from theflow.schema.defaults.base import BaseDefaultsConfig
from theflow.schema.defaults.utils import DefaultsDataclassField
from theflow.schema.features.base import BaseFeatureConfig
from theflow.schema.features.utils import gbm_defaults_config_registry
from theflow.schema.utils import theflow_dataclass


@DeveloperAPI
@theflow_dataclass
class GBMDefaultsConfig(BaseDefaultsConfig):
    binary: BaseFeatureConfig = DefaultsDataclassField(
        feature_type=BINARY, defaults_registry=gbm_defaults_config_registry
    )

    category: BaseFeatureConfig = DefaultsDataclassField(
        feature_type=CATEGORY, defaults_registry=gbm_defaults_config_registry
    )

    number: BaseFeatureConfig = DefaultsDataclassField(
        feature_type=NUMBER, defaults_registry=gbm_defaults_config_registry
    )


@DeveloperAPI
class GBMDefaultsField(schema_utils.DictMarshmallowField):
    def __init__(self):
        super().__init__(GBMDefaultsConfig)
