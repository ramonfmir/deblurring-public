#!/bin/sh

cd deblurring
export PYTHONPATH=.
pip install --upgrade pip
pip install -r scripts/requirements_azure.txt
python3 cnn_denoiser/train.py --run restart --num_iter 1 --model_name moussaka
./scripts/slack_file_upload output.txt "#performance-metrics"
rm output.txt
