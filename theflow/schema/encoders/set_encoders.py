from typing import List

from theflow.api_annotations import DeveloperAPI
from theflow.constants import SET
from theflow.schema import utils as schema_utils
from theflow.schema.encoders.base import BaseEncoderConfig
from theflow.schema.encoders.utils import register_encoder_config
from theflow.schema.metadata import ENCODER_METADATA
from theflow.schema.utils import theflow_dataclass


@DeveloperAPI
@register_encoder_config("embed", SET)
@theflow_dataclass
class SetSparseEncoderConfig(BaseEncoderConfig):
    @staticmethod
    def module_name():
        return "SetSparseEncoder"

    type: str = schema_utils.ProtectedString(
        "embed",
        description=ENCODER_METADATA["SetSparseEncoder"]["type"].long_description,
    )

    dropout: float = schema_utils.FloatRange(
        default=0.0,
        min=0,
        max=1,
        description="Dropout probability for the embedding.",
        parameter_metadata=ENCODER_METADATA["SetSparseEncoder"]["dropout"],
    )

    activation: str = schema_utils.ActivationOptions(
        description="The default activation function that will be used for each layer.",
        parameter_metadata=ENCODER_METADATA["SetSparseEncoder"]["activation"],
    )

    representation: str = schema_utils.StringOptions(
        ["dense", "sparse"],
        default="dense",
        description="The representation of the embedding. Either dense or sparse.",
        parameter_metadata=ENCODER_METADATA["SetSparseEncoder"]["representation"],
    )

    vocab: List[str] = schema_utils.List(
        default=None,
        description="Vocabulary of the encoder",
        parameter_metadata=ENCODER_METADATA["SetSparseEncoder"]["vocab"],
    )

    use_bias: bool = schema_utils.Boolean(
        default=True,
        description="Whether the layer uses a bias vector.",
        parameter_metadata=ENCODER_METADATA["SetSparseEncoder"]["use_bias"],
    )

    bias_initializer: str = schema_utils.InitializerOptions(
        default="zeros",
        description="Initializer to use for the bias vector.",
        parameter_metadata=ENCODER_METADATA["SetSparseEncoder"]["bias_initializer"],
    )

    weights_initializer: str = schema_utils.InitializerOptions(
        description="Initializer to use for the weights matrix.",
        parameter_metadata=ENCODER_METADATA["SetSparseEncoder"]["weights_initializer"],
    )

    embedding_size: int = schema_utils.PositiveInteger(
        default=50,
        description="The maximum embedding size, the actual size will be min(vocabulary_size, embedding_size) for "
        "dense representations and exactly vocabulary_size for the sparse encoding, where vocabulary_size "
        "is the number of different strings appearing in the training set in the input column (plus 1 for "
        "the unknown token placeholder <UNK>).",
        parameter_metadata=ENCODER_METADATA["SetSparseEncoder"]["embedding_size"],
    )

    embeddings_on_cpu: bool = schema_utils.Boolean(
        default=False,
        description="By default embedding matrices are stored on GPU memory if a GPU is used, as it allows for faster "
        "access, but in some cases the embedding matrix may be too large. This parameter forces the "
        "placement of the embedding matrix in regular memory and the CPU is used for embedding lookup, "
        "slightly slowing down the process as a result of data transfer between CPU and GPU memory.",
        parameter_metadata=ENCODER_METADATA["SetSparseEncoder"]["embeddings_on_cpu"],
    )

    embeddings_trainable: bool = schema_utils.Boolean(
        default=True,
        description="If true embeddings are trained during the training process, if false embeddings are fixed. It "
        "may be useful when loading pretrained embeddings for avoiding finetuning them. This parameter "
        "has effect only when representation is dense as sparse one-hot encodings are not trainable.",
        parameter_metadata=ENCODER_METADATA["SetSparseEncoder"]["embeddings_trainable"],
    )

    pretrained_embeddings: str = schema_utils.String(
        default=None,
        allow_none=True,
        description="By default dense embeddings are initialized randomly, but this parameter allows to specify a "
        "path to a file containing embeddings in the GloVe format. When the file containing the "
        "embeddings is loaded, only the embeddings with labels present in the vocabulary are kept, "
        "the others are discarded. If the vocabulary contains strings that have no match in the "
        "embeddings file, their embeddings are initialized with the average of all other embedding plus "
        "some random noise to make them different from each other. This parameter has effect only if "
        "representation is dense.",
        parameter_metadata=ENCODER_METADATA["SetSparseEncoder"]["pretrained_embeddings"],
    )

    output_size: int = schema_utils.PositiveInteger(
        default=10,
        description="If output_size is not already specified in fc_layers this is the default output_size that will "
        "be used for each layer. It indicates the size of the output of a fully connected layer.",
        parameter_metadata=ENCODER_METADATA["SetSparseEncoder"]["output_size"],
    )

    norm: str = schema_utils.StringOptions(
        ["batch", "layer"],
        default=None,
        allow_none=True,
        description="The default norm that will be used for each layer.",
        parameter_metadata=ENCODER_METADATA["SetSparseEncoder"]["norm"],
    )

    norm_params: dict = schema_utils.Dict(
        default=None,
        description="Parameters used if norm is either `batch` or `layer`.",
        parameter_metadata=ENCODER_METADATA["SetSparseEncoder"]["norm_params"],
    )

    num_fc_layers: int = schema_utils.NonNegativeInteger(
        default=0,
        description="This is the number of stacked fully connected layers that the input to the feature passes "
        "through. Their output is projected in the feature's output space.",
        parameter_metadata=ENCODER_METADATA["SetSparseEncoder"]["num_fc_layers"],
    )

    fc_layers: List[dict] = schema_utils.DictList(  # TODO (Connor): Add nesting logic for fc_layers
        default=None,
        description="List of dictionaries containing the parameters for each fully connected layer.",
        parameter_metadata=ENCODER_METADATA["SetSparseEncoder"]["fc_layers"],
    )
