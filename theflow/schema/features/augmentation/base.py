from theflow.api_annotations import DeveloperAPI
from theflow.schema import utils as schema_utils
from theflow.schema.utils import theflow_dataclass


@DeveloperAPI
@theflow_dataclass
class BaseAugmentationConfig(schema_utils.BaseMarshmallowConfig):
    """Base class for augmentation."""

    type: str
