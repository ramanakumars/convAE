import os
import random
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf#; tf.compat.v1.disable_eager_execution()
from keras.layers import Layer, Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, \
    Conv3D, UpSampling3D, Conv3DTranspose, MaxPool3D, Activation, BatchNormalization, LeakyReLU, \
    Dropout, MaxPool2D, UpSampling2D, Lambda, AveragePooling2D, Add, AveragePooling3D, Cropping2D, \
    RepeatVector, Permute
from tensorflow.keras import activations
from keras.models import Model
from keras import metrics
from keras import backend as K
from keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import Adam, RMSprop, Adagrad, SGD
from keras.losses import KLDivergence
from keras import regularizers
import netCDF4 as nc
from keras.models import load_model
#tf.executing_eagerly()

MODEL_SAVE_FOLDER = os.environ.get('MODEL_SAVE_FOLDER', 'models/')
L2_LOSS = 1.e-3