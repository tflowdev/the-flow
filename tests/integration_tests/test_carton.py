# Copyright (c) 2023 Predibase, Inc., 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import asyncio
import os
import platform
from typing import List, Union

import numpy as np
import pandas as pd
import pytest
import torch

from theflow.api import The FlowModel
from theflow.constants import BATCH_SIZE, NAME, PREDICTIONS, TRAINER
from theflow.utils.carton_utils import export_carton
from tests.integration_tests.utils import (
    binary_feature,
    category_feature,
    generate_data,
    LocalTestBackend,
    number_feature,
)


@pytest.mark.skipif(platform.system() == "Windows", reason="Carton is not supported on Windows")
def test_carton_torchscript(csv_filename, tmpdir):
    data_csv_path = os.path.join(tmpdir, csv_filename)

    # Configure features to be tested:
    bin_str_feature = binary_feature()
    input_features = [
        bin_str_feature,
        # binary_feature(),
        number_feature(),
        category_feature(encoder={"vocab_size": 3}),
        # TODO: future support
        # sequence_feature(vocab_size=3),
        # text_feature(vocab_size=3),
        # vector_feature(),
        # image_feature(image_dest_folder),
        # audio_feature(audio_dest_folder),
        # timeseries_feature(),
        # date_feature(),
        # h3_feature(),
        # set_feature(vocab_size=3),
        # bag_feature(vocab_size=3),
    ]
    output_features = [
        bin_str_feature,
        # binary_feature(),
        number_feature(),
        category_feature(decoder={"vocab_size": 3}, output_feature=True),
        # TODO: future support
        # sequence_feature(vocab_size=3),
        # text_feature(vocab_size=3),
        # set_feature(vocab_size=3),
        # vector_feature()
    ]
    backend = LocalTestBackend()
    config = {
        "input_features": input_features,
        "output_features": output_features,
        TRAINER: {"epochs": 2, BATCH_SIZE: 128},
    }

    # Generate training data
    training_data_csv_path = generate_data(input_features, output_features, data_csv_path)

    # Convert bool values to strings, e.g., {'Yes', 'No'}
    df = pd.read_csv(training_data_csv_path)
    false_value, true_value = "No", "Yes"
    df[bin_str_feature[NAME]] = df[bin_str_feature[NAME]].map(lambda x: true_value if x else false_value)
    df.to_csv(training_data_csv_path)

    # Train The Flow (Pythonic) model:
    theflow_model = The FlowModel(config, backend=backend)
    theflow_model.train(
        dataset=training_data_csv_path,
        skip_save_training_description=True,
        skip_save_training_statistics=True,
        skip_save_model=True,
        skip_save_progress=True,
        skip_save_log=True,
        skip_save_processed_input=True,
    )

    # Obtain predictions from Python model
    preds_dict, _ = theflow_model.predict(dataset=training_data_csv_path, return_type=dict)

    # Create graph inference model (Torchscript) from trained The Flow model.
    carton_path = os.path.join(tmpdir, "carton")
    export_carton(theflow_model, carton_path)

    import cartonml as carton

    # Load the carton model
    # See https://pyo3.rs/v0.20.0/ecosystem/async-await#a-note-about-asynciorun for why we wrap it
    # in another function
    async def load():
        return await carton.load(carton_path)

    loop = asyncio.get_event_loop()
    carton_model = loop.run_until_complete(load())

    def to_input(s: pd.Series) -> Union[List[str], torch.Tensor]:
        if s.dtype == "object":
            return np.array(s.to_list())
        return s.to_numpy().astype(np.float32)

    df = pd.read_csv(training_data_csv_path)
    inputs = {name: to_input(df[feature.column]) for name, feature in theflow_model.model.input_features.items()}

    # See https://pyo3.rs/v0.20.0/ecosystem/async-await#a-note-about-asynciorun for why we wrap it
    # in another function
    async def infer(inputs):
        return await carton_model.infer(inputs)

    outputs = loop.run_until_complete(infer(inputs))

    # Compare results from Python trained model against Carton
    assert len(preds_dict) == len(outputs)
    for feature_name, feature_outputs_expected in preds_dict.items():
        assert feature_name in outputs

        output_values_expected = feature_outputs_expected[PREDICTIONS]
        output_values = outputs[feature_name]
        if output_values.dtype.type in {np.string_, np.str_}:
            # Strings should match exactly
            assert np.all(output_values == output_values_expected), f"feature: {feature_name}, output: predictions"
        else:
            assert np.allclose(output_values, output_values_expected), f"feature: {feature_name}, output: predictions"
