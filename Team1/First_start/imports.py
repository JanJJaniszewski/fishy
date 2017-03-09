# Keras
from keras import backend as K
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.datasets import cifar10
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.preprocessing import image

# Others
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
from IPython.display import Image
from IPython.core.display import HTML 
from multi_gpu import make_parallel
from matplotlib import pyplot
from scipy.misc import toimage
import tensorflow as tf

# Commands to use Keras with Tesnors Flow
K.set_image_dim_ordering('tf')

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


print('Imports imported')