from theflow.api_annotations import DeveloperAPI
from theflow.constants import (
    AUDIO,
    BAG,
    BINARY,
    CATEGORY,
    DATE,
    H3,
    IMAGE,
    NUMBER,
    SEQUENCE,
    SET,
    TEXT,
    TIMESERIES,
    VECTOR,
)
from theflow.schema import utils as schema_utils
from theflow.schema.defaults.base import BaseDefaultsConfig
from theflow.schema.defaults.utils import DefaultsDataclassField
from theflow.schema.features.base import BaseFeatureConfig
from theflow.schema.utils import theflow_dataclass


@DeveloperAPI
@theflow_dataclass
class ECDDefaultsConfig(BaseDefaultsConfig):
    audio: BaseFeatureConfig = DefaultsDataclassField(feature_type=AUDIO)

    bag: BaseFeatureConfig = DefaultsDataclassField(feature_type=BAG)

    binary: BaseFeatureConfig = DefaultsDataclassField(feature_type=BINARY)

    category: BaseFeatureConfig = DefaultsDataclassField(feature_type=CATEGORY)

    date: BaseFeatureConfig = DefaultsDataclassField(feature_type=DATE)

    h3: BaseFeatureConfig = DefaultsDataclassField(feature_type=H3)

    image: BaseFeatureConfig = DefaultsDataclassField(feature_type=IMAGE)

    number: BaseFeatureConfig = DefaultsDataclassField(feature_type=NUMBER)

    sequence: BaseFeatureConfig = DefaultsDataclassField(feature_type=SEQUENCE)

    set: BaseFeatureConfig = DefaultsDataclassField(feature_type=SET)

    text: BaseFeatureConfig = DefaultsDataclassField(feature_type=TEXT)

    timeseries: BaseFeatureConfig = DefaultsDataclassField(feature_type=TIMESERIES)

    vector: BaseFeatureConfig = DefaultsDataclassField(feature_type=VECTOR)


@DeveloperAPI
class ECDDefaultsField(schema_utils.DictMarshmallowField):
    def __init__(self):
        super().__init__(ECDDefaultsConfig)
