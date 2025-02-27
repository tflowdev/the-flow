import os
import time

import numpy as np
import pandas as pd
import pytest
import torch
from packaging.version import parse as parse_version

if parse_version(pd.__version__) > parse_version("1.5.3"):
    pytest.skip(allow_module_level=True)


from theflow.api import The FlowModel
from theflow.benchmarking.profiler import The FlowProfiler
from theflow.constants import BATCH_SIZE, TRAINER


@pytest.mark.skipif(
    parse_version(pd.__version__) > parse_version("1.5.3"),
    reason="experiment_impact_tracker package is incompatible with pandas 2.0",
)
def test_theflow_profiler(tmpdir):
    @The FlowProfiler(tag="test_function", output_dir=tmpdir, use_torch_profiler=False, logging_interval=0.1)
    def func(duration):
        time.sleep(duration)
        x = torch.Tensor(2, 3)
        y = torch.rand(2, 3)
        torch.add(x, y)

    train_df = pd.DataFrame(np.random.normal(0, 1, size=(100, 3)), columns=["input_1", "input_2", "output_1"])
    eval_df = pd.DataFrame(np.random.normal(0, 1, size=(20, 3)), columns=["input_1", "input_2", "output_1"])

    config = {
        "input_features": [{"name": "input_1", "type": "number"}, {"name": "input_2", "type": "number"}],
        "output_features": [{"name": "output_1", "type": "number"}],
        "combiner": {"type": "concat", "output_size": 14},
        TRAINER: {"epochs": 1, BATCH_SIZE: 128},
    }

    model = The FlowModel(config=config, backend="local")

    with The FlowProfiler(tag="profile_1", output_dir=tmpdir, use_torch_profiler=False, logging_interval=0.1):
        model.train(
            dataset=train_df,
            output_directory=tmpdir,
            skip_save_training_description=True,
            skip_save_training_statistics=True,
            skip_save_model=True,
            skip_save_progress=True,
            skip_save_log=True,
            skip_save_processed_input=True,
        )

    assert os.path.exists(os.path.join(tmpdir, "system_resource_usage", "profile_1", "run_0.json"))

    with The FlowProfiler(tag="profile_2", output_dir=tmpdir, use_torch_profiler=True, logging_interval=0.1):
        model.evaluate(dataset=eval_df)
        func(0.1)

    assert os.path.exists(os.path.join(tmpdir, "system_resource_usage", "profile_2", "run_0.json"))
    assert os.path.exists(os.path.join(tmpdir, "torch_ops_resource_usage", "profile_2", "run_0.json"))

    func(0.25)
    func(0.5)
    assert set(os.listdir(os.path.join(tmpdir, "system_resource_usage", "test_function"))) == {
        "run_0.json",
        "run_1.json",
        "run_2.json",
    }
