#!/bin/sh

function run() {
  cd deblurring
  export PYTHONPATH=.
  pip3 install --upgrade pip
  pip3 install -r requirements_azure.txt
  python3 data/dataset_manipulation/generate_dataset_same_dimensions.py
  python3 deblurrer/cnn_autoencoder/train.py --run restart --num_iter 1 --model_name tutorial_cnn
  ./slack_file_upload output.txt "#performance-metrics"
  rm -rf data/4000unlabeledLP_same_dims_scaled
  rm output.txt
}
