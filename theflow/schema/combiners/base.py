from theflow.api_annotations import DeveloperAPI
from theflow.schema import utils as schema_utils
from theflow.schema.utils import theflow_dataclass


@DeveloperAPI
@theflow_dataclass
class BaseCombinerConfig(schema_utils.BaseMarshmallowConfig):
    """Base combiner config class."""

    type: str
