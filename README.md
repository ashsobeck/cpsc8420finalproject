# CPSC 8420 Final Project

## Introduction
CNN’s take images and use filters to classify them into different categories. While training a CNN is a novel concept today, tweaking the parameters and structure of a CNN is a problem that depends on different variables. What we want to focus on is also the data that is fed into a CNN. Since these networks are so powerful we want to test the limit of their abilities. Image recognition systems have been in place for a while and will only get better. As the amount of images on the internet and devices will only grow storing them could pose as a challenge. With this project we hope to see if training CNN’s on compressed images can be as effective as normal ones. This could help reduce data if compressed images can be regenerated, and classified as well as uncompressed ones.

## Method
We are reducing these images via PCA. We plan to train 2 models with the regular dataset, and the CIFAR-10 dataset that has been compressed. We will compare the Training and Testing Loss and Accuracy.


## Running
Run the python module: model.py will compress the images and train the two models

