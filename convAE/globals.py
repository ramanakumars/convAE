import os
import random
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf; tf.compat.v1.disable_eager_execution()
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
from keras import regularizers
import netCDF4 as nc
from keras.models import load_model
#tf.executing_eagerly()

MODEL_SAVE_FOLDER = os.environ.get('MODEL_SAVE_FOLDER', 'models/')
L2_LOSS = 1.e-3

# A function to compute the value of latent space
def compute_latent(x):
    mu, sigma = x
    batch = K.shape(mu)[0]
    dim = K.int_shape(mu)[1]
    eps = K.random_normal(shape=(batch,dim))
    return mu + K.exp(sigma/2)*eps

def compute_latent2(mu, sigma=-4):
    batch = K.shape(mu)[0]
    dim = K.int_shape(mu)[1]
    eps = K.random_normal(shape=(batch,dim))
    return mu + K.exp(sigma/2)*eps

def encoder_layers2D(enc_inp, conv_filt, conv_act, hidden):
    enc_layers = []
    enc_layers.append(Conv2D(4, (2, 2), padding='valid', strides=(1,1), 
                        activation=conv_act, name='enc_conv_0')(enc_inp))
    enc_layers.append(AveragePooling2D((2,2))(enc_layers[-1]))
    enc_layers.append(BatchNormalization(name='enc_batch_norm_0')(enc_layers[-1]))

    enc_layers.append(Conv2D(128, (2, 2), padding='valid', strides=(1,1), 
                        activation=conv_act, name='enc_conv_1')(enc_layers[-1]))
    enc_layers.append(AveragePooling2D((2,2))(enc_layers[-1]))
    enc_layers.append(BatchNormalization(name='enc_batch_norm_1')(enc_layers[-1]))

    enc_layers.append(Conv2D(conv_filt, (2, 2), padding='valid', strides=(1,1), 
                        activation=conv_act, name='enc_conv_2')(enc_layers[-1]))
    enc_layers.append(AveragePooling2D((2,2))(enc_layers[-1]))
    enc_layers.append(BatchNormalization(name='enc_batch_norm_2')(enc_layers[-1]))
    

    enc_layers.append(Conv2D(conv_filt, (2, 2), padding='valid', strides=(1,1), 
                        activation=conv_act, name='enc_conv_3')(enc_layers[-1]))
    enc_layers.append(AveragePooling2D((2,2))(enc_layers[-1]))
    enc_layers.append(BatchNormalization(name='enc_batch_norm_3')(enc_layers[-1]))

    enc_layers.append(Conv2D(conv_filt, (2, 2), padding='valid', strides=(1,1), 
                        activation=conv_act, name='enc_conv_4')(enc_layers[-1]))
    enc_layers.append(AveragePooling2D((2,2))(enc_layers[-1]))
    enc_layers.append(BatchNormalization(name='enc_batch_norm_4')(enc_layers[-1]))
    
    #enc_p  = AveragePooling3D(pool_size=(2,2,1), name='enc_pool')(enc_b4)

    nconv = len(hidden)
    for i in range(nconv):
        if i < nconv-1:
            acti = conv_act
        else:
            acti = None
        enc_layers.append(Conv2D(hidden[i], (1,1), padding='valid', strides=(1,1), 
                            activation=acti, name=f'enc_conv_dense_{i}')(enc_layers[-1]))
        enc_layers.append(BatchNormalization(name=f'enc_bn_dense_{i}')(enc_layers[-1]))

    return enc_layers

def decoder_layers2D(dec_inp, conv_filt, conv_act, hidden):
    dec_layers = [dec_inp]
    nconv = len(hidden)
    for i in range(nconv):
        dec_layers.append(Conv2D(hidden[::-1][i], (1,1), padding='valid', strides=(1,1), 
                            activation=conv_act, name=f'dec_conv_dense_{i}')(dec_layers[-1]))
        dec_layers.append(BatchNormalization(name=f'dec_bn_dense_{i}')(dec_layers[-1]))

    #dec_layers.append(Conv2DTranspose(conv_filt, (2, 2), padding='valid', strides=(1,1), 
    #                    activation=conv_act, name='dec_conv_0')(dec_layers[-1]))

    dec_layers.append(UpSampling2D((2,2))(dec_layers[-1]))
    dec_layers.append(Conv2DTranspose(conv_filt, (2, 2), padding='valid', strides=(1,1), 
                        activation=conv_act, name='dec_conv_1')(dec_layers[-1]))
    dec_layers.append(BatchNormalization(name='dec_batch_norm_0')(dec_layers[-1]))

    #dec_layers.append(Conv2DTranspose(conv_filt, (2, 2), padding='valid', strides=(1,1), 
    #                    activation=conv_act, name='dec_conv_2')(dec_layers[-1]))
    #dec_layers.append(UpSampling2D((2,2))(dec_layers[-1]))
    #dec_layers.append(BatchNormalization(name='dec_batch_norm_1')(dec_layers[-1]))

    dec_layers.append(UpSampling2D((2,2))(dec_layers[-1]))
    dec_layers.append(Conv2DTranspose(conv_filt, (3, 3), padding='valid', strides=(1,1), 
                        activation=conv_act, name='dec_conv_3')(dec_layers[-1]))
    dec_layers.append(BatchNormalization(name='dec_batch_norm_2')(dec_layers[-1]))

    dec_layers.append(UpSampling2D((2,2))(dec_layers[-1]))
    dec_layers.append(Conv2DTranspose(128, (2, 2), padding='valid', strides=(1,1), 
                        activation=conv_act, name='dec_conv_4')(dec_layers[-1]))
    dec_layers.append(BatchNormalization(name='dec_batch_norm_3')(dec_layers[-1]))

    dec_layers.append(UpSampling2D((2,2))(dec_layers[-1]))
    dec_layers.append(Conv2DTranspose(4, (2, 2), padding='valid', strides=(1,1), 
                        activation=conv_act, name='dec_conv_5')(dec_layers[-1]))
    dec_layers.append(BatchNormalization(name='dec_batch_norm_4')(dec_layers[-1]))

    dec_layers.append(UpSampling2D((2,2))(dec_layers[-1]))
    dec_layers.append(Conv2DTranspose(3, (2, 2), padding='valid', strides=(1, 1), 
                        activation='sigmoid', name='dec_conv_6')(dec_layers[-1]))

    dec_layers.append(Cropping2D(((3,4), (3,4)))(dec_layers[-1]))

    return dec_layers

def encoder_layers2D_256(enc_inp, conv_filt, conv_act, hidden):
    enc_layers = []
    enc_layers.append(Conv2D(4, (2, 2), padding='valid', strides=(1,1), 
                        activation=conv_act, name='enc_conv_0')(enc_inp))
    enc_layers.append(AveragePooling2D((2,2))(enc_layers[-1]))
    enc_layers.append(BatchNormalization(name='enc_batch_norm_0')(enc_layers[-1]))

    enc_layers.append(Conv2D(128, (2, 2), padding='valid', strides=(1,1), 
                        activation=conv_act, name='enc_conv_1')(enc_layers[-1]))
    enc_layers.append(AveragePooling2D((2,2))(enc_layers[-1]))
    enc_layers.append(BatchNormalization(name='enc_batch_norm_1')(enc_layers[-1]))

    enc_layers.append(Conv2D(conv_filt, (2, 2), padding='valid', strides=(1,1), 
                        activation=conv_act, name='enc_conv_2')(enc_layers[-1]))
    enc_layers.append(AveragePooling2D((2,2))(enc_layers[-1]))
    enc_layers.append(BatchNormalization(name='enc_batch_norm_2')(enc_layers[-1]))
    

    enc_layers.append(Conv2D(conv_filt, (2, 2), padding='valid', strides=(1,1), 
                        activation=conv_act, name='enc_conv_3')(enc_layers[-1]))
    enc_layers.append(AveragePooling2D((2,2))(enc_layers[-1]))
    enc_layers.append(BatchNormalization(name='enc_batch_norm_3')(enc_layers[-1]))

    enc_layers.append(Conv2D(conv_filt, (2, 2), padding='valid', strides=(1,1), 
                        activation=conv_act, name='enc_conv_4')(enc_layers[-1]))
    enc_layers.append(AveragePooling2D((2,2))(enc_layers[-1]))
    enc_layers.append(BatchNormalization(name='enc_batch_norm_4')(enc_layers[-1]))
    
    enc_layers.append(Conv2D(conv_filt, (2, 2), padding='valid', strides=(1,1), 
                        activation=conv_act, name='enc_conv_5')(enc_layers[-1]))
    enc_layers.append(AveragePooling2D((2,2))(enc_layers[-1]))
    enc_layers.append(BatchNormalization(name='enc_batch_norm_5')(enc_layers[-1]))
    
    #enc_p  = AveragePooling3D(pool_size=(2,2,1), name='enc_pool')(enc_b4)

    nconv = len(hidden)
    for i in range(nconv):
        if i < nconv-1:
            acti = conv_act
        else:
            acti = None
        enc_layers.append(Conv2D(hidden[i], (1,1), padding='valid', strides=(1,1), 
                            activation=acti, name=f'enc_conv_dense_{i}')(enc_layers[-1]))
        enc_layers.append(BatchNormalization(name=f'enc_bn_dense_{i}')(enc_layers[-1]))

    return enc_layers

def decoder_layers2D_256(dec_inp, conv_filt, conv_act, hidden):
    dec_layers = [dec_inp]
    nconv = len(hidden)
    for i in range(nconv):
        dec_layers.append(Conv2D(hidden[::-1][i], (1,1), padding='valid', strides=(1,1), 
                            activation=conv_act, name=f'dec_conv_dense_{i}')(dec_layers[-1]))
        dec_layers.append(BatchNormalization(name=f'dec_bn_dense_{i}')(dec_layers[-1]))

    #dec_layers.append(Conv2DTranspose(conv_filt, (2, 2), padding='valid', strides=(1,1), 
    #                    activation=conv_act, name='dec_conv_0')(dec_layers[-1]))

    dec_layers.append(UpSampling2D((2,2))(dec_layers[-1]))
    dec_layers.append(Conv2DTranspose(conv_filt, (2, 2), padding='valid', strides=(1,1), 
                        activation=conv_act, name='dec_conv_0')(dec_layers[-1]))
    dec_layers.append(BatchNormalization(name='dec_batch_norm_0')(dec_layers[-1]))

    dec_layers.append(UpSampling2D((2,2))(dec_layers[-1]))
    dec_layers.append(Conv2DTranspose(conv_filt, (2, 2), padding='valid', strides=(1,1), 
                        activation=conv_act, name='dec_conv_1')(dec_layers[-1]))
    dec_layers.append(BatchNormalization(name='dec_batch_norm_1')(dec_layers[-1]))

    #dec_layers.append(Conv2DTranspose(conv_filt, (2, 2), padding='valid', strides=(1,1), 
    #                    activation=conv_act, name='dec_conv_2')(dec_layers[-1]))
    #dec_layers.append(UpSampling2D((2,2))(dec_layers[-1]))
    #dec_layers.append(BatchNormalization(name='dec_batch_norm_1')(dec_layers[-1]))

    dec_layers.append(UpSampling2D((2,2))(dec_layers[-1]))
    dec_layers.append(Conv2DTranspose(conv_filt, (3, 3), padding='valid', strides=(1,1), 
                        activation=conv_act, name='dec_conv_2')(dec_layers[-1]))
    dec_layers.append(BatchNormalization(name='dec_batch_norm_2')(dec_layers[-1]))

    dec_layers.append(UpSampling2D((2,2))(dec_layers[-1]))
    dec_layers.append(Conv2DTranspose(128, (2, 2), padding='valid', strides=(1,1), 
                        activation=conv_act, name='dec_conv_3')(dec_layers[-1]))
    dec_layers.append(BatchNormalization(name='dec_batch_norm_3')(dec_layers[-1]))

    dec_layers.append(UpSampling2D((2,2))(dec_layers[-1]))
    dec_layers.append(Conv2DTranspose(4, (2, 2), padding='valid', strides=(1,1), 
                        activation=conv_act, name='dec_conv_4')(dec_layers[-1]))
    dec_layers.append(BatchNormalization(name='dec_batch_norm_4')(dec_layers[-1]))

    dec_layers.append(UpSampling2D((2,2))(dec_layers[-1]))
    dec_layers.append(Conv2DTranspose(3, (2, 2), padding='valid', strides=(1, 1), 
                        activation='sigmoid', name='dec_conv_6')(dec_layers[-1]))

    dec_layers.append(Cropping2D(((3,4), (3,4)))(dec_layers[-1]))

    return dec_layers

