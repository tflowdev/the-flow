from theflow.api_annotations import DeveloperAPI
from theflow.constants import FILL_WITH_CONST, H3, MISSING_VALUE_STRATEGY_OPTIONS, PREPROCESSING
from theflow.schema import utils as schema_utils
from theflow.schema.features.preprocessing.base import BasePreprocessingConfig
from theflow.schema.features.preprocessing.utils import register_preprocessor
from theflow.schema.metadata import FEATURE_METADATA
from theflow.schema.utils import theflow_dataclass


@DeveloperAPI
@register_preprocessor(H3)
@theflow_dataclass
class H3PreprocessingConfig(BasePreprocessingConfig):
    missing_value_strategy: str = schema_utils.StringOptions(
        MISSING_VALUE_STRATEGY_OPTIONS,
        default=FILL_WITH_CONST,
        allow_none=False,
        description="What strategy to follow when there's a missing value in an h3 column",
        parameter_metadata=FEATURE_METADATA[H3][PREPROCESSING]["missing_value_strategy"],
    )

    fill_value: int = schema_utils.PositiveInteger(
        default=576495936675512319,
        allow_none=False,
        description="The value to replace missing values with in case the missing_value_strategy is fill_with_const",
        parameter_metadata=FEATURE_METADATA[H3][PREPROCESSING]["fill_value"],
    )

    computed_fill_value: int = schema_utils.PositiveInteger(
        default=576495936675512319,
        allow_none=False,
        description="The internally computed fill value to replace missing values with in case the "
        "missing_value_strategy is fill_with_mode or fill_with_mean",
        parameter_metadata=FEATURE_METADATA[H3][PREPROCESSING]["computed_fill_value"],
    )
