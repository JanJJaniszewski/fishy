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
from keras.wrappers.scikit_learn import KerasClassifier

# Others
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
from IPython.display import Image
from IPython.core.display import HTML 
from matplotlib import pyplot
import os 
from scipy import ndimage
from subprocess import check_output
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
import pprint

# Commands to use Keras with Tesnors Flow
K.set_image_dim_ordering('tf')

print('Imports imported')
