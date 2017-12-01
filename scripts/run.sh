#!/bin/sh

cd deblurring
export PYTHONPATH=.
pip3 install --upgrade pip
pip3 install -r scripts/requirements_azure.txt
python3 cnn_denoiser/train.py --run restart --num_iter 1 --model_name tutorial_cnn
./scripts/slack_file_upload output.txt "#performance-metrics"
rm output.txt
