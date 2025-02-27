"""Train a model from entirely synthetic data."""

import logging
import tempfile

import yaml

from theflow.api import The FlowModel
from theflow.data.dataset_synthesizer import build_synthetic_dataset_df

config = yaml.safe_load(
    """
input_features:
    - name: Pclass (new)
      type: category

output_features:
    - name: Survived
      type: binary

"""
)

df = build_synthetic_dataset_df(120, config)
model = The FlowModel(config, logging_level=logging.INFO)

with tempfile.TemporaryDirectory() as tmpdir:
    model.train(dataset=df, output_directory=tmpdir)
