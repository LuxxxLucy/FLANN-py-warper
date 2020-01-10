# FLANN-py-warpper
 A sci-kit like warper for FLANN k-nearest neighbour classifier in python

__FLANN__: (FLANN Fast Library for Approximate Nearest Neighbors) is a library for performing fast approximate nearest neighbor searches in high dimensional spaces.

Here I provide a python scikit like warper.

* Some re-use of checking functions is from scikit.
* Only the KNN classifier is implemented, also weighted distance. Although that the regressor should be easy to implement in the same way.

## usage

Same usage as the Scikit KNN classifier

## Effect

See the example of usage in `example.ipynb` I test the K=1 KNN classifier in MNIST.

Exact KNN -> Approximation of FLANN

__Accuracy__: 13  -> 15 ~ 16

__Run time__: 13.7s ± 1.2s -> 1.01s ± 59.1ms

The run time is tested in 10 rounds.

## Install

Please just download the file and use as local python modules.

See the example of usage in `example.ipynb`

Also, the installation of the FLANN through pip `!pip install pyflann-py3`
