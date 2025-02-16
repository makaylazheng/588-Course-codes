# 588-Course-codes

## Diffusion
This repository contains two notebooks demonstrating the forward and backward processes of diffusion models applied to the MNIST and CIFAR-10 datasets.

- `diffusers_mnist.ipynb` and `diffusers_cifar.ipynb`

The notebooks illustrate how noise is gradually added to the images during the forward process and then removed during the reverse process to generate new samples. 
The reverse (backward) process utilizes the **trained noise predictor models** stored in the corresponding checkpoints folders. 

The Python (.py) files in this repository contain the training code used to develop these noise predictor models.

- `diffusion_mnist.py` and `diffusion_cifar.py`
