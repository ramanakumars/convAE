#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from convAE import ConvVAE_DEC
from convAE import ConvVAE
import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np

conv_filt     = 512
hidden        = [128, 16]
beta          = 0.5
learning_rate = 1.e-4
sigma0        = -4.
conv_act      = 'tanh'

batch_size=32

vae_dec = ConvVAE_DEC(sigma0, beta, conv_act, conv_filt, hidden)
vae_dec.n_centroid = 10
vae_dec.create_model()
vae_dec.add_loss_funcs()

learning_rate = 1.e-5
vae_dec.compile(learning_rate=learning_rate, optimizer='Adam', decay=0.)
#vae.create_lr_scheduler(learning_rate, 0.95, 30)
#vae_dec.load_vae_weights(vae.get_savefolder())
vae_dec.set_initial_positions_test()

#vae_dec.train(data, epochs=300, batch_size=batch_size)

vae_dec.save()

