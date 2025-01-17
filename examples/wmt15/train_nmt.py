"""Sample theflow training code for training an NMT model (en -> fr) on WMT15 (https://www.statmt.org/wmt15/).

The dataset is rather large (8GB), which can take several minutes to preprocess.
"""

import logging
import shutil

from theflow.api import The FlowModel
from theflow.datasets import wmt15

# clean out prior results
shutil.rmtree("./results", ignore_errors=True)

# Download and prepare the dataset
training_set = wmt15.load()

model = The FlowModel(config="./config_small.yaml", logging_level=logging.INFO)

(
    train_stats,  # dictionary containing training statistics
    preprocessed_data,  # tuple The Flow Dataset objects of pre-processed training data
    output_directory,  # location of training results stored on disk
) = model.train(dataset=training_set, experiment_name="simple_experiment", model_name="simple_model")
