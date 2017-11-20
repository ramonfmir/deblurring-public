# Deblurring

## Table of Contents
1. Run instructions (How to deblur images)
2. Project Inspiration and Introduction
3. Technologies
4. How the **Blurring** works
5. How the **De-Blurring** works
6. Team Members

## Run Intructions
### Running locally
To deblur a directory of images, first create a destination directory where you want the deblurred images to go. Then run:
* `python3 deblur_images.py  /path/to/your/blurred/images  /path/to/destination`

We have added the deblurred version of the images in `final_test` that you sent us in the directory `final_test_deblurred`. You can also see the results in the validation images in `data/100clean`.

# Project Inspiration and Introduction
There are two main themes in this project:
1. The Blurrer - Applies artificial, realistic blurs to images to allow us to train the De-Blurrer.
2. The De-Blurrer - Receives a blurry image and attempts to produce the original.

We are doing this for the following reasons:
* Huawei's 2017 license plate deblurring competition.
* Imperial College 3rd year Software Engineering group project.
* We like big acronyms (DCGAN, ML, DPN, CNN...).

# Technologies
Our final model was trained by Tensorflow GPU on a single K80 Graphics Card for 18 hours in an Azure NC12 VM.
For Blurring we mainly used OpenCV Python.
For model trainng and testing evaluation we used Tensorboard.

# How the De-Blurring works
## Image Processing
The network requires that images have dimensions 270x90. Input images must be scaled before propagation. We used to keep the aspect ratio (and fill the gaps with black padding) but we found that this was not necessary. Though the spatial relationships between features is distorted, the network is trained on numberplates taken with a variety of different camera perspectives so is well equipped to deal with these distortions.
The image is then converted to grey scale. This has the advantages:
1. Only one colour channel is required, reducing training and evaluation time, and network complexity.
2. Numberplates of different colours all look the same, so there is less to learn.

## Model Architecture ([Our model](https://github.com/qvks/deblurring/blob/ready_for_sumission/cnn_denoiser/model_definitions/networks/moussaka.py))
(defined in cnn_denoiser/model_definitions/networks/moussaka.py)
The model consists of three convolutional networks, and three transposed convolutional networks. We called it `moussaka`, a traditional Greek dish that consists of several layers.
Before each layer (except the first layer) we apply dropout with a rate of 0.5 to avoid overfitting. After each layer we apply ReLU.
### Encoding (Convolutional)
The first three layers apply kernels to the image as it propagates. Typically, we have observed that the first layer identifies edges, and so on. At each layer, the image is padded out with black borders so that the resultant image would have the same dimensions if the kernel stride was 1.
However, in the first two layers the stride is 3 (in the horizontal and vertical directions), reducing the image width and height by 1/3, and the area by 1/9. Thus, the third convolutional layer is fed a 30x10 image. It has a stride of 1, so we arrive at the 'code layer' with an image of size 30x10.
### Decoding (Transposed Convolutional)
These three layers mirror the first three convolutional layers. The first transposed convolutional layer has a stride of 1 and is fed images from the code layer.
The proceeding two transposed convolutions have a stride of 3 - hence, the final (hopefully deblurred) image is returned to a size of 270x90. The last layer, though it takes many input channels, only outputs one image. To this image we apply the tanh activation function to constrict the output and yield the final (and hopefully deblurred) image.

## Model Training ([Our Training Script](https://github.com/qvks/deblurring/blob/ready_for_sumission/cnn_denoiser/train.py))
Initially, we pretrain the model as a Deep Belief Network.
We then train in constant size batches selected from the shuffled dataset. Once we have iterated over the dataset once, we reshuffle and start again.
To each batch, and for each image in the batch, we derive the blurred image from the original image using our blur function. Thus, the cost of the network output is the mean square of the different between the output (deblurred) image and the original image.
We use an Adam Optimiser to train the weights on the aforementioned cost with a learning rate that decays exponentially over time.

To run the trianing script we run the following:
`python3 cnn_denoiser/train.py --run restart --num_iter 99999 --model moussaka`

## Model Evaluation
When training our model we used Tensorboard to evaluate its progress. Every 10 batches we save an image from just one channel of the propagation described above. We also record the average error of the batch and the current learning rate which are plotted as graphs.
The weights are saved to a folder with the same periodicity. We can then load these weights into a local model and run them on 100 blurred images from real life. This gives us the best idea of how well the model is doing.

# Team Members
* Ramon Fernandez Mir (@ramonfmir)
* Christopher Hawkes (@chawkn)
* Robert Holland (@RobbieHolland)
* Ruiao Hu (@Dochu1)
* Rachel Mekhtieva Lee (@qvks)
* Aristomenis Papadopoulos (@arisPapadop)
