from typing import Optional

from theflow.api_annotations import DeveloperAPI
from theflow.schema import utils as schema_utils
from theflow.schema.combiners.base import BaseCombinerConfig
from theflow.schema.combiners.common_transformer_options import CommonTransformerConfig
from theflow.schema.combiners.utils import register_combiner_config
from theflow.schema.metadata import COMBINER_METADATA
from theflow.schema.utils import theflow_dataclass


@DeveloperAPI
@register_combiner_config("transformer")
@theflow_dataclass
class TransformerCombinerConfig(BaseCombinerConfig, CommonTransformerConfig):
    """Parameters for transformer combiner."""

    type: str = schema_utils.ProtectedString(
        "transformer",
        description=COMBINER_METADATA["transformer"]["type"].long_description,
    )

    reduce_output: Optional[str] = schema_utils.ReductionOptions(
        default="mean",
        description="Strategy to use to aggregate the output of the transformer.",
        parameter_metadata=COMBINER_METADATA["transformer"]["reduce_output"],
    )
