python3 deblurring/data/dataset_manipulation/generate_dataset_same_dimensions.py
python3 deblurring/deblurrer/cnn_autoencoder/train.py --run restart --num_iter 10 --model_name tutorial_cnn
./slack_file_upload output.txt "#performance-metrics"
