from .globals import *
from .layers import *
import re

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


class BaseVariationalAE():
    def __init__(self, sigma0=0., beta=1.e-3, conv_act='relu', conv_filt=512, hidden=[128, 16]):
        self.conv_act  = conv_act
        self.sig0      = sigma0
        self.kl_beta   = beta
        self.conv_filt = conv_filt
        self.hidden    = hidden
        self.nconv     = len(hidden)

    def compile(self, learning_rate=0.0001, optimizer='Adam', decay=0.0):
        if optimizer=='Adam':
            opt = Adam(learning_rate=learning_rate, decay=decay)
        elif optimizer=='Adagrad':
            opt = Adagrad(learning_rate=learning_rate)
        elif optimizer=='SGD':
            opt = SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True, decay=decay)
        elif optimizer=='RMSprop':
            opt = RMSprop(learning_rate=learning_rate)

        self.ae.compile(optimizer=opt)
        self.create_name()
    
    def create_lr_scheduler(self, learning_rate, lr_drop, lr_drop_frequency):
        self.learning_rate = learning_rate
        self.lr_drop       = lr_drop
        self.lr_drop_frequency = lr_drop_frequency
        
        self.lr_scheduler = LearningRateScheduler(self.rate_decay)

    def rate_decay(self, epoch):
        initial = self.learning_rate
        drop    = self.lr_drop
        nepoch  = self.lr_drop_frequency

        lrate = initial*(drop**np.floor((1+epoch)/nepoch))

        return lrate

    def get_recon_loss(self):
        recon_loss = K.mean(K.sum(K.square(self.input - self.output), axis=(1,2)), axis=(1))

        return tf.reduce_mean(recon_loss)

    def get_kl_loss(self):
        mui,sigi, z  = self.encoder(self.input)
        mup, sigp, zp = self.encoder(self.output)

        sig0 = self.sig0*K.ones_like(mui)
        
        kl = - 1 - sigi + sig0 + (K.square(mui-mup) + K.exp(sigi))/K.exp(sig0)
        kl = K.mean(kl, axis=-1)
        kl *= 0.5*self.kl_beta

        return tf.reduce_mean(kl)

    def get_cluster_loss(self):
        mui,sigi, z  = self.encoder(self.input)

        batch_size = tf.shape(z)[0]

        gamma_layer = self.ae.get_layer('gamma')
        Z, ZMean, ZLogVar = gamma_layer.get_z_vals([mui, sigi, z])

        theta_tensor3, u_tensor3, lambda_tensor3 = gamma_layer.get_tensors(batch_size)

        gamma = self.cluster(self.input)
        gamma_t = tf.repeat(tf.expand_dims(self.gamma, axis=2), tf.shape(theta_tensor3)[2], axis=2)

        print(f"gamma_t: {gamma_t.get_shape()}, gamma: {gamma.get_shape()}  tensor: {theta_tensor3.get_shape()}")

        a = 0.5*K.sum(gamma*K.sum(K.log(lambda_tensor3)+
                        K.exp(ZLogVar)/lambda_tensor3+
                        K.square(ZMean-u_tensor3)/lambda_tensor3, axis=2),axis=(1,2))
        d = K.sum(K.log(K.mean(theta_tensor3,axis=2)/gamma)*gamma, axis=(1,2))

        clust_loss = tf.reduce_mean(a-d)

        return clust_loss

    def train(self, data, epochs=300, batch_size=10, checkpoint_freq=10):

        savesfolder = self.get_savefolder()

        self.nepochs = epochs

        if not os.path.exists(savesfolder):
            os.mkdir(savesfolder)

        if not hasattr(self, 'starting_epoch'):
            self.starting_epoch=0

        save_freq = int(np.ceil(len(data)*0.9/batch_size))*checkpoint_freq
        print(f"Saving checkpoints every {save_freq} batches")

        # create checkpoints
        checkpoint_filepath = '%s/checkpoint-{epoch:05d}.hdf5'%savesfolder
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_filepath, verbose=1,
            save_weights_only=True,
            save_freq=save_freq, initial_epoch=self.starting_epoch)

        print(f"Training {self.name}")
        if hasattr(self, 'lr_scheduler'):
            print("Using Learning Rate Scheduler")
            self.history = self.ae.fit(data, epochs=epochs, validation_split=0.1, initial_epoch=self.starting_epoch,
                            callbacks=[self.lr_scheduler, checkpoint_callback], validation_freq=5, 
                            batch_size=batch_size, shuffle=True)
        else:
            self.history = self.ae.fit(data, epochs=epochs, validation_split=0.1, initial_epoch=self.starting_epoch,
                            callbacks=[checkpoint_callback], validation_freq=5, batch_size=batch_size,
                            shuffle=True)
    def create_name(self):
        hidden_name = ''
        for layeri in self.hidden:
            hidden_name += '_%d'%layeri
        self.name = 'vae_%dls_%dconv%d%s_%s'%(self.latent_dim, self.nconv, self.conv_filt, hidden_name, self.conv_act)

        if self.batch_norm:
            self.name += "_batchnorm"
        if self.batch_norm2:
            self.name += "_batchnorm2"
        
        self.name += "_sig%1.0f_beta_%1.0e"%(self.ae.sig0, self.ae.kl_beta)

    def save(self):
        savesfolder = self.get_savefolder()
        self.ae.encoder.save_weights(savesfolder+"encoderw.h5")
        self.ae.decoder.save_weights(savesfolder+"decoderw.h5")

        if hasattr(self, 'cluster'):
            self.cluster.save_weights(savesfolder+"clusterw.h5")
    
    def load(self):
        savesfolder = self.get_savefolder()
        self.ae.encoder.load_weights(savesfolder+"encoderw.h5")
        self.ae.decoder.load_weights(savesfolder+"decoderw.h5")
        if hasattr(self, 'cluster'):
            self.cluster.load_weights(savesfolder+"clusterw.h5")
        #self.ae.load_weights(savesfolder+"VAEw.h5")

    def load_last_checkpoint(self):
        savesfolder = self.get_savefolder()
        checkpoints = glob("%s/checkpoint-*.hdf5"%savesfolder)
        if len(checkpoints)>0:
            last_checkpoint = natural_sort(checkpoints)[-1]
            self.ae.load_weights(last_checkpoint)

            self.starting_epoch = int(last_checkpoint.split('/')[-1].split('-')[1].split('.')[0])
            print(f"Loaded checkpoint for epoch {self.starting_epoch}")
        else:
            print("No checkpoints found!")


    def get_savefolder(self):
        self.savesfolder =  f'{MODEL_SAVE_FOLDER}/{self.conv_act}/models-{self.name}/'
        return self.savesfolder

    def add_loss_funcs(self):
        r_loss   = self.get_recon_loss()
        kl_loss  = self.get_kl_loss()

        # sum of all three losses
        loss = r_loss + kl_loss# + c_loss

        self.ae.add_loss(loss)
        self.ae.add_metric(r_loss, aggregation='mean', name='mse')
        self.ae.add_metric(kl_loss, aggregation='mean', name='kl')
