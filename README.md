# Image-Classification
In this repository some projects associated with image recognition is presented.

Firstly, Mnist and FashionMnist classes are presented in Utils for defining some important functions
It is to be noted that FashionMnist is the Child of Mnist class.
Several models such as FC, CNN and LSTM are utilized for training these datasets.
As these datasets are simple to analyze, the accuracy of the aforementioned models is more than 95%.

The other part of this repository is associated with CIFAR10 dataset, which is trained implementing a CNN.
As this code is run by my personal laptop, it is not possible to build a sophisticated CNN and therefore, the 
accuracy of the presented CNN model in cifar10_CNN.py is 71%. with the aid of data augmentation and powerfull
computers with standard GPUs, it is possible to increase the accuracy of a CNN up to 99%. More details can be
found in the following benchmark website.

https://benchmarks.ai/cifar-10