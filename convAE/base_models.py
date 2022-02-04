from .globals import *

class GammaLayer(Layer):
    def __init__(self, latent_dim, n_centroid, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim#/npixels)
        self.n_centroid = n_centroid
        
    def build(self, input_shape):
        theta_init  = np.ones((1, 1, self.n_centroid))/self.n_centroid
        u_init      = np.random.random((1, self.latent_dim, self.n_centroid))#-0.5
        lambda_init = np.random.random((1, self.latent_dim, self.n_centroid))
        
        # define the variables used by this layer
        self.theta_p  = tf.Variable(theta_init, trainable=True, shape=(1, 1, self.n_centroid,), name="pi", dtype=tf.float32)
        self.u_p      = tf.Variable(u_init, trainable=True, shape=(1, self.latent_dim, self.n_centroid), name="u", dtype=tf.float32)
        self.lambda_p = tf.Variable(lambda_init, trainable=True, shape=(1, self.latent_dim, self.n_centroid),name="lambda", dtype=tf.float32)

        super().build(input_shape)

    def get_z_vals(self, x, only_z=False):
        n_centroid = self.n_centroid; latent_dim = self.latent_dim
        mu, sig, z = x
        

        # reshape the latent values
        Z = tf.transpose(K.repeat(z, n_centroid),perm=[0,2,1])
        Z = Reshape((self.latent_dim, n_centroid))(Z)

        if not only_z:
            z_mean_t = tf.transpose(K.repeat(mu,n_centroid),perm=[0,2,1])
            z_log_var_t = tf.transpose(K.repeat(sig,n_centroid),[0,2,1])
            
            Zmu  = Reshape((self.latent_dim, n_centroid))(z_mean_t)
            Zsig = Reshape((self.latent_dim, n_centroid))(z_log_var_t)

            return Z, Zmu, Zsig
        else:
            return Z

    def get_tensors(self, batch_size):
        u_tensor3 = tf.repeat(self.u_p, batch_size, axis=0)#self.u_p*K.ones((batch_size, self.latent_dim, n_centroid))
        lambda_tensor3 = tf.repeat(self.lambda_p, batch_size, axis=0)
        thetai = tf.repeat(self.theta_p, self.latent_dim, axis=1)
        theta_tensor3 = tf.repeat(\
                                    thetai,\
                                  batch_size, axis=0)#/K.sum(thetai, axis=(2), keepdims=True)

        return theta_tensor3, u_tensor3, lambda_tensor3

    def call(self, x, training=None):
        n_centroid = self.n_centroid; latent_dim = self.latent_dim
        mu, sig, z = x

        batch_size = tf.shape(z)[0]

        # reshape the latent values
        Z = self.get_z_vals(x, only_z=True)

        # build the tensors for calculating gamma
        theta_tensor3, u_tensor3, lambda_tensor3 = self.get_tensors(batch_size)

        # categorical distribution
        p_c   = K.sum(K.log(theta_tensor3), axis=1)# - 0.5*K.log(lambda_tensor3*2.*np.pi), axis=1)

        # normal distribution
        p_z_c = K.sum(-K.square(Z - u_tensor3)/(2*lambda_tensor3), axis=1)

        # p(c|z) = p(c)*p(z|c) = exp( log(p(c)) + log(p(z|c)) )
        p_c_z = K.exp(p_c + p_z_c) + 1.e-10

        # get gamma=p(c|z)/sum(p(c|z))
        gamma = p_c_z/K.sum(p_c_z,axis=-1,keepdims=True)

        return gamma#self.gamma_t


class GammaLayerConv(Layer):
    def __init__(self, latent_dim, n_centroid, npixels, **kwargs):
        super().__init__(**kwargs)
        self.npixels    = npixels
        self.latent_dim = int(latent_dim/npixels)
        self.n_centroid = n_centroid
        
    def build(self, input_shape):
        theta_init  = np.ones((1, self.npixels, 1, self.n_centroid))/self.n_centroid
        u_init      = np.random.random((1, self.npixels, self.latent_dim, self.n_centroid))
        lambda_init = np.random.random((1, self.npixels, self.latent_dim, self.n_centroid))
        
        # define the variables used by this layer
        self.theta_p  = tf.Variable(theta_init, trainable=True, shape=(1, self.npixels, 1, self.n_centroid,), name="pi", dtype=tf.float32)
        self.u_p      = tf.Variable(u_init, trainable=True, shape=(1, self.npixels, self.latent_dim, self.n_centroid), name="u", dtype=tf.float32)
        self.lambda_p = tf.Variable(lambda_init, trainable=True, shape=(1,self.npixels, self.latent_dim, self.n_centroid),name="lambda", dtype=tf.float32)

        super().build(input_shape)

    def get_tensors(self, batch_size):
        u_tensor3 = tf.repeat(self.u_p, batch_size, axis=0)#*#K.ones((batch_size, self.npixels, self.latent_dim, n_centroid))
        lambda_tensor3 = tf.repeat(self.lambda_p, batch_size, axis=0)
        theta_tensor3 = tf.repeat(\
                                    tf.repeat(self.theta_p, self.latent_dim, axis=2), \
                                batch_size, axis=0)

        return theta_tensor3, u_tensor3, lambda_tensor3

    def get_z_vals(self, x, only_z=False):
        n_centroid = self.n_centroid; latent_dim = self.latent_dim; npixels = self.npixels
        mu, sig, z = x
        

        # reshape the latent values
        Z = tf.transpose(K.repeat(z, n_centroid),perm=[0,2,1])
        Z = Reshape((self.npixels, self.latent_dim, n_centroid))(Z)

        if not only_z:
            z_mean_t = tf.transpose(K.repeat(mu,n_centroid),perm=[0,2,1])
            z_log_var_t = tf.transpose(K.repeat(sig,n_centroid),[0,2,1])
            
            Zmu  = Reshape((self.npixels, self.latent_dim, n_centroid))(z_mean_t)
            Zsig = Reshape((self.npixels, self.latent_dim, n_centroid))(z_log_var_t)

            return Z, Zmu, Zsig
        else:
            return Z

    def call(self, x, training=None):
        n_centroid = self.n_centroid; latent_dim = self.latent_dim
        mu, sig, z = x

        Z = self.get_z_vals(x, only_z=True)
        
        batch_size = tf.shape(z)[0]

        # build the tensors for calculating gamma
        theta_tensor3, u_tensor3, lambda_tensor3 = self.get_tensors(batch_size)

        #p_c_z = K.exp(K.sum(a - b - c ,axis=(2)) )+1e-10
        p_c   = K.sum(K.log(theta_tensor3), axis=2)# - 0.5*K.log(lambda_tensor3*2.*np.pi), axis=1)

        # normal distribution
        p_z_c = K.sum(-K.square(Z - u_tensor3)/(2*lambda_tensor3), axis=2)

        # p(c|z) = p(c)*p(z|c) = exp( log(p(c)) + log(p(z|c)) )
        p_c_z = K.exp(p_c + p_z_c) + 1.e-10

        # get gamma=p(c|z)/sum(p(c|z))
        gamma = p_c_z/K.sum(p_c_z,axis=-1,keepdims=True)

        self.gamma = p_c_z/K.sum(p_c_z,axis=-1,keepdims=True)

        return self.gamma

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

    def train(self, data, epochs=300, batch_size=10):

        savesfolder = self.get_savefolder()

        self.nepochs = epochs

        if not os.path.exists(savesfolder):
            os.mkdir(savesfolder)

        if not hasattr(self, 'starting_epoch'):
            self.starting_epoch=0

        save_freq = int(np.ceil(len(data)*0.9/batch_size))*2
        print(f"Saving checkpoints every {save_freq} batches")

        # create checkpoints
        checkpoint_filepath = '%s/checkpoint-{epoch:02d}.hdf5'%savesfolder
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_filepath, verbose=1,
            save_best_only=False, save_weights_only=False,
            save_freq=save_freq, initial_epoch=self.starting_epoch)

        print(f"Training {self.name}")
        if hasattr(self, 'lr_scheduler'):
            print("Using Learning Rate Scheduler")
            self.history = self.ae.fit(data, epochs=epochs, validation_split=0.1,
                            callbacks=[self.lr_scheduler, checkpoint_callback], validation_freq=5, 
                            batch_size=batch_size, shuffle=True)
        else:
            self.history = self.ae.fit(data, epochs=epochs, validation_split=0.1, 
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
        last_checkpoint = sorted(glob("%s/checkpoint-*.h5"%savesfolder))[-1]
        self.ae.load_weights(last_checkpoint)

        self.starting_epoch = int(last_checkpoint.split('-')[1].split('.')[0])

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
