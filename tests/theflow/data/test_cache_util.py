import copy
import uuid
from typing import List
from unittest import mock

import pytest

from theflow.constants import INPUT_FEATURES, OUTPUT_FEATURES
from theflow.data.cache.util import calculate_checksum
from theflow.schema.model_types.base import ModelConfig
from theflow.types import FeatureConfigDict, ModelConfigDict
from theflow.utils.misc_utils import merge_dict


def _gen_config(input_features: List[FeatureConfigDict]) -> ModelConfigDict:
    return {INPUT_FEATURES: input_features, OUTPUT_FEATURES: [{"name": "out1", "type": "binary"}]}


@pytest.mark.parametrize(
    "input_features,diff,expected",
    [
        (
            [
                {
                    "name": "in1",
                    "type": "text",
                    "encoder": {"type": "parallel_cnn"},
                }
            ],
            [
                {
                    "encoder": {"type": "stacked_cnn"},
                }
            ],
            True,
        ),
        (
            [
                {
                    "name": "in1",
                    "type": "text",
                    "preprocessing": {"cache_encoder_embeddings": True},
                    "encoder": {"type": "bert"},
                }
            ],
            [
                {
                    "encoder": {"type": "distilbert"},
                }
            ],
            False,
        ),
    ],
)
def test_calculate_checksum(input_features: List[FeatureConfigDict], diff: List[FeatureConfigDict], expected: bool):
    config = _gen_config(input_features)

    diff_features = [merge_dict(f, df) for f, df in zip(input_features, diff)]
    diff_config = _gen_config(diff_features)

    mock_dataset = mock.Mock()
    mock_dataset.checksum = uuid.uuid4().hex

    assert (
        calculate_checksum(mock_dataset, ModelConfig.from_dict(config).to_dict())
        == calculate_checksum(mock_dataset, ModelConfig.from_dict(diff_config).to_dict())
    ) == expected


def test_proc_col_checksum_consistency():
    """Tests that proc_col is equal if checksum are equal."""
    config_dict1 = {
        "input_features": [{"name": "txt1", "type": "text", "encoder": {"type": "bert"}}],
        "output_features": [{"name": "bin1", "type": "binary"}],
    }
    config1 = ModelConfig.from_dict(config_dict1)

    config_dict2 = copy.deepcopy(config_dict1)
    config_dict2["input_features"][0]["preprocessing"] = {
        "tokenizer": "bert",
    }
    config2 = ModelConfig.from_dict(config_dict2)

    mock_dataset = mock.Mock()
    mock_dataset.checksum = uuid.uuid4().hex
    assert calculate_checksum(mock_dataset, config1.to_dict()) == calculate_checksum(mock_dataset, config2.to_dict())

    for if1, if2 in zip(config1.input_features, config2.input_features):
        assert if1.name == if2.name
        assert if1.proc_column == if2.proc_column

    for of1, of2 in zip(config1.output_features, config2.output_features):
        assert of1.name == of2.name
        assert of1.proc_column == of2.proc_column


def test_proc_col_checksum_consistency_same_preprocessing_different_types():
    """Tests that proc_col is different if preprocessing and names are the same but types are different."""
    config = {
        "input_features": [
            # Same name, different types, same preprocessing
            {"name": "num1", "type": "number", "preprocessing": {"missing_value_strategy": "fill_with_mode"}},
            {"name": "num2", "type": "category", "preprocessing": {"missing_value_strategy": "fill_with_mode"}},
        ],
        "output_features": [
            {"name": "num3", "type": "number", "preprocessing": {"missing_value_strategy": "fill_with_mode"}}
        ],
    }
    config = ModelConfig.from_dict(config)

    assert config.input_features[0].proc_column != config.input_features[1].proc_column


@pytest.mark.distributed
def test_checksum_determinism(ray_cluster_2cpu):
    """Tests that checksums are deterministic across different processes (no unordered hash maps)."""
    import ray

    # Generate a lot of features so the probability of a reordering of feature sets is very high.
    config = {
        INPUT_FEATURES: [{"name": f"in{i}", "type": "number"} for i in range(100)],
        OUTPUT_FEATURES: [{"name": "out1", "type": "binary"}],
    }
    config = ModelConfig.from_dict(config)

    mock_dataset = mock.Mock()
    mock_dataset.checksum = uuid.uuid4().hex

    @ray.remote(max_calls=1)
    def calculate_checksum_remote(dataset, config):
        return calculate_checksum(dataset, config)

    # Run each checksum calculation as a remote function so it gets its own Python interpreter, as
    # the hash function in Python is deterministic within a process, but not between different processes.
    # See: https://docs.python.org/3/reference/datamodel.html#object.__hash__
    checksum1 = ray.get(calculate_checksum_remote.remote(mock_dataset, config.to_dict()))
    checksum2 = ray.get(calculate_checksum_remote.remote(mock_dataset, config.to_dict()))
    assert checksum1 == checksum2
