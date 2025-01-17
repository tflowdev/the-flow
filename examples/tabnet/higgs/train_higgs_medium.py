import logging

from theflow.api import The FlowModel
from theflow.datasets import higgs

model = The FlowModel(
    config="medium_config.yaml",
    logging_level=logging.INFO,
)

higgs_df = higgs.load()
model.train(dataset=higgs_df, experiment_name="higgs_medium", model_name="higgs_tabnet_medium")
