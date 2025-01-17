from typing import Optional

from theflow.api_annotations import DeveloperAPI
from theflow.constants import MODEL_ECD, SEQUENCE
from theflow.schema import utils as schema_utils
from theflow.schema.combiners.base import BaseCombinerConfig
from theflow.schema.combiners.sequence_concat import MAIN_SEQUENCE_FEATURE_DESCRIPTION
from theflow.schema.combiners.utils import register_combiner_config
from theflow.schema.encoders.base import BaseEncoderConfig
from theflow.schema.encoders.utils import EncoderDataclassField
from theflow.schema.metadata import COMBINER_METADATA
from theflow.schema.utils import theflow_dataclass

"""
SEQUENCE encoders that always return 2D [batch_size, hidden_size] tensors, regardless of how they are parameterized.
These should never be used with modules that expect 3D tensors, such as the SequenceCombiner.
"""
_2D_SEQUENCE_ENCODERS = ["embed"]


@DeveloperAPI
@register_combiner_config("sequence")
@theflow_dataclass
class SequenceCombinerConfig(BaseCombinerConfig):
    """Parameters for sequence combiner."""

    type: str = schema_utils.ProtectedString(
        "sequence",
        description=COMBINER_METADATA["sequence"]["type"].long_description,
    )

    main_sequence_feature: Optional[str] = schema_utils.String(
        default=None,
        allow_none=True,
        description=MAIN_SEQUENCE_FEATURE_DESCRIPTION,
        parameter_metadata=COMBINER_METADATA["sequence"]["main_sequence_feature"],
    )

    encoder: BaseEncoderConfig = EncoderDataclassField(
        MODEL_ECD,
        feature_type=SEQUENCE,
        default="parallel_cnn",
        description="Encoder to apply to `main_sequence_feature`. The encoder must produce"
        " a tensor of size [batch_size, sequence_length, hidden_size]",
        blocklist=_2D_SEQUENCE_ENCODERS,
    )

    reduce_output: Optional[str] = schema_utils.ReductionOptions(
        default=None,
        description="Strategy to use to aggregate the embeddings of the items of the set.",
        parameter_metadata=COMBINER_METADATA["sequence"]["reduce_output"],
    )
