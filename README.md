# Deblurring

[![Build Status](https://travis-ci.com/qvks/deblurring.svg?token=fTTycZszr1xwSx4deq7e&branch=master)](https://travis-ci.com/qvks/deblurring) [![codecov](https://codecov.io/gh/ramonfmir/deblurring/branch/master/graph/badge.svg?token=54QoaHZuqI)](https://codecov.io/gh/ramonfmir/deblurring)


Project used for:

* Huawei's 2017 license plate deblurring competition.
* Imperial College 3rd year Software Engineering group project.

Members:

* Ramon Fernandez Mir (@ramonfmir)
* Christopher Hawkes (@chawkn)
* Robert Holland (@RobbieHolland)
* Ruiao Hu (@Dochu1)
* Rachel Mekhtieva Lee (@qvks)
* Aristomenis Papadopoulos (@arisPapadop)

## Prerequisites

Follow these steps to make sure that you can run our code locally:

1. Clone this repository.

2. Install Python 3.6 or later. Simply run `sudo apt-get install python3.6`.

3. Upgrade `pip3` and install the rest of the requirements by typing `pip3 install -r requirements.txt`.


## Running locally

Not so exciting at the moment. You can do a few things though. An example would be running:

* `python3 data/dataset_manipulation/generate_dataset_same_dimensions.py`
* `python3 deblurrer/cnn_denoise_experiment/train.py restart`

This should train our neural network.
