#!/usr/bin/env python

# # Simple Model Training Example
#
# This example is the API example for this The Flow command line example
# (https://theflow-ai.github.io/theflow-docs/latest/examples/titanic/).

# Import required libraries
import logging
import os
import shutil

import yaml

from theflow.api import The FlowModel
from theflow.datasets import titanic

# clean out prior results
shutil.rmtree("./results", ignore_errors=True)

# Download and prepare the dataset
training_set, test_set, _ = titanic.load(split=True)

config = yaml.safe_load(
    """
input_features:
    - name: Pclass
      type: category
    - name: Sex
      type: category
    - name: Age
      type: number
      preprocessing:
          missing_value_strategy: fill_with_mean
    - name: SibSp
      type: number
    - name: Parch
      type: number
    - name: Fare
      type: number
      preprocessing:
          missing_value_strategy: fill_with_mean
    - name: Embarked
      type: category

output_features:
    - name: Survived
      type: binary

"""
)

# Define The Flow model object that drive model training
model = The FlowModel(config=config, logging_level=logging.INFO)

# initiate model training
(
    train_stats,  # dictionary containing training statistics
    preprocessed_data,  # tuple The Flow Dataset objects of pre-processed training data
    output_directory,  # location of training results stored on disk
) = model.train(
    dataset=training_set, experiment_name="simple_experiment", model_name="simple_model", skip_save_processed_input=True
)

# list contents of output directory
print("contents of output directory:", output_directory)
for item in os.listdir(output_directory):
    print("\t", item)

# batch prediction
model.predict(test_set, skip_save_predictions=False)
