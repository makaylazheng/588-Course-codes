# 588-Course-codes

## Diffusion
This repository contains two notebooks demonstrating the forward and backward processes of diffusion models applied to the MNIST and CIFAR-10 datasets.

- `diffusers_mnist.ipynb` and `diffusers_cifar.ipynb`

The notebooks illustrate how noise is gradually added to the images during the forward process and then removed during the reverse process to generate new samples. 
The reverse (backward) process utilizes the **trained noise predictor models** stored in the corresponding checkpoints folders. 

The Python (.py) files in this repository contain the training code used to develop these noise predictor models.

- `diffusion_mnist.py` and `diffusion_cifar.py`


## FCNN
This folder contains a notebook demonstrating the construction and training of fully connected neural networks on MNIST dataset. The implementation contains several techniqus introduced in the lecture, including mini-batch, early stopping, L2 regularization, dropout and Xavier initialization. It also includes a tuning example of batch size and dropout rate. 

## CNN
This folder contains a notebook demonstrating how to construct and train CNN on CIFAR-10 dataset. It includes a simple CNN and a deep CNN, and compares their performance on the same dataset. 

## RNN
This folder contains a notebook of RNN and a folder of data. In the notebook, we construct and train a RNN to predict the language based on the given words.

