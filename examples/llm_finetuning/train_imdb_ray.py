import logging
import os

import yaml

from theflow.api import The FlowModel

config = yaml.safe_load(
    """
input_features:
  - name: review
    type: text

    encoder:
      type: auto_transformer
      pretrained_model_name_or_path: bigscience/bloom-3b
      trainable: true
      adapter:
        type: lora

output_features:
  - name: sentiment
    type: category

trainer:
  batch_size: 4
  epochs: 3

backend:
  type: ray
  trainer:
    use_gpu: true
    strategy:
      type: deepspeed
      zero_optimization:
        stage: 3
        offload_optimizer:
          device: cpu
          pin_memory: true
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
    dataset="theflow://imdb",
    experiment_name="imdb_sentiment",
    model_name="bloom3b",
)

# list contents of output directory
print("contents of output directory:", output_directory)
for item in os.listdir(output_directory):
    print("\t", item)
