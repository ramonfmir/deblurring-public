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

## Usage

To deblur images, run the following command:

`python3 deblur_images /path/to/corrupted /path/to/clean`

This deblur all the images in the directory `path/to/corrupted` and put them in the directory `path/to/clean`.
