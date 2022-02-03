from .globals import *
from .base_models import BaseVariationalAE, GammaLayer, GammaLayerConv

class ColorAE(BaseVariationalAE):
    def create_model(self, conv_act='tanh', pool=False):
        hidden     = self.hidden
        conv_filt  = self.conv_filt
        self.conv_act  = conv_act

        ''' ENCODER '''
        input_shape = (128, 128, 3)

        # Constructing encoder
        self.input = encoder_input = Input(shape=input_shape, name='input')

        reshape_layer1 = Reshape(target_shape=(*input_shape, 1), name='3d_reshape')(self.input)

        # convolution part
        enc_layers = encoder_layers(reshape_layer1, conv_filt, conv_act, hidden)
        input_conv = enc_layers[-1]
        latent_space = Flatten(name='latent')(enc_layers[-1])
        
        # Build the encoder
        self.encoder = Model(encoder_input, latent_space, name='encoder')

        self.encoder.summary()

        self.z = latent_space
        
        ''' DECODER '''
        # Take the convolution shape to be used in the decoder
        conv_shape = K.int_shape(input_conv)
        
        # Constructing decoder
        dec_inps = []
        decoder_input = Input(shape=K.int_shape(latent_space)[1:], name='dec_inp')
        dec_inps.append(decoder_input)
        
        dec3 = Reshape(conv_shape[1:])(decoder_input)

        
        dec_layers = decoder_layers(dec3, conv_filt, conv_act, hidden)
        
        decoder_output = Reshape(target_shape=input_shape, name='3d_reshape')(dec_layers[-1])
        # Build the decoder
        self.decoder = Model(decoder_input, decoder_output, name='decoder')
        self.decoder.summary()

        self.output = self.decoder(self.encoder(encoder_input))
        
        # Build the VAE
        self.ae = Model(encoder_input, self.output, name='AE')

        self.ae.encoder = self.encoder
        self.ae.decoder = self.decoder

        self.ae.summary()
        
    def create_name(self):
        hidden_name = ''
        for layeri in self.hidden:
            hidden_name += '_%d'%layeri
        self.name = 'colorae_%dconv%d%s'%(self.nconv, self.conv_filt, hidden_name)

        print(self.name)
    
    def add_loss_funcs(self):

        z = self.encoder(self.input)
        zp = self.encoder(self.output)

        z_mse_loss = K.sum(K.square(z - zp), axis=1)

        z_loss   = tf.nn.compute_average_loss(z_mse_loss)
        r_loss   = self.get_recon_loss()

        # sum of all three losses
        loss = r_loss + z_loss# + c_loss

        self.ae.add_loss(loss)
        self.ae.add_metric(r_loss, aggregation='mean', name='mse')
        self.ae.add_metric(z_loss, aggregation='mean', name='z')

class ColorVAE(BaseVariationalAE):
    def create_model(self, sigma0=0., beta=1.e-3, conv_act='tanh', pool=False):
        hidden     = self.hidden
        conv_filt  = self.conv_filt
        latent_dim = self.latent_dim
        batch_norm = self.batch_norm
        batch_norm2 = self.batch_norm2
        self.conv_act  = conv_act
        self.pool   = pool

        ''' ENCODER '''
        input_shape = (128, 128, 3)

        # Constructing encoder
        self.input = encoder_input = Input(shape=input_shape, name='input')

        reshape_layer1 = Reshape(target_shape=(*input_shape, 1), name='3d_reshape')(self.input)
        
        enc_layers = []

        # convolution part
        '''
        for i in range(self.nconv):
            conv_layer = Conv3D(conv_filt, (2**(3-i), 2**(3-i),1), padding='valid', strides=(2,2,1), 
                                activation=conv_act, name=f'enc_conv_{i}')
            if i==0:
                enc_layers.append(conv_layer(reshape_layer1))
            else:
                if batch_norm:
                    enc_layers.append(conv_layer(enc_layers[-1]))
                else:
                    enc_layers.append(conv_layer(enc_layers[-1]))
            #if (i%2==1):
            #    enc_layers.append(MaxPool3D(pool_size=(2,2,1), padding='same', name=f'enc_pool_{i}')(enc_layers[-1]))
            if batch_norm:
                enc_layers.append(BatchNormalization(name=f'enc_batch_norm_{i}')(enc_layers[-1]))

        if pool:
            enc_layers.append(AveragePooling3D(pool_size=(2,2,1), padding='valid', name=f'enc_pool')(enc_layers[-1]))

        enc_c1 = Conv3D(4, (4, 4, 1), padding='valid', strides=(2,2,1), 
                            activation=conv_act, name='enc_conv_0')(reshape_layer1)
        enc_b1 = BatchNormalization(name='enc_batch_norm_0')(enc_c1)

        enc_c2 = Conv3D(32, (3, 3, 1), padding='valid', strides=(2,2,1), 
                            activation=conv_act, name='enc_conv_1')(enc_b1)
        enc_b2 = BatchNormalization(name='enc_batch_norm_1')(enc_c2)

        enc_c3 = Conv3D(conv_filt, (3, 3, 1), padding='valid', strides=(2,2,1), 
                            activation=conv_act, name='enc_conv_2')(enc_b2)
        enc_b3 = BatchNormalization(name='enc_batch_norm_2')(enc_c3)
        

        enc_c4 = Conv3D(conv_filt, (2, 2, 1), padding='valid', strides=(2,2,1), 
                            activation=conv_act, name='enc_conv_3')(enc_b3)
        enc_b4 = BatchNormalization(name='enc_batch_norm_3')(enc_c4)

        enc_c5 = Conv3D(conv_filt, (2, 2, 1), padding='valid', strides=(2,2,1), 
                            activation=conv_act, name='enc_conv_4')(enc_b4)
        enc_b5 = BatchNormalization(name='enc_batch_norm_4')(enc_c5)
        
        #enc_p  = AveragePooling3D(pool_size=(2,2,1), name='enc_pool')(enc_b4)

        enc_layers = [enc_b5]

        for i in range(self.nconv):
            if i < self.nconv-1:
                acti = conv_act
            else:
                acti = None
            enc_layers.append(Conv3D(hidden[i], (1,1,1), padding='valid', strides=(1,1,1), 
                                     activation=acti, name=f'enc_conv_dense_{i}')(enc_layers[-1]))
            enc_layers.append(BatchNormalization(name=f'enc_bn_dense_{i}')(enc_layers[-1]))

        input_conv = enc_layers[-1]
        '''

        enc_layers = encoder_layers(reshape_layer1, conv_filt, conv_act, hidden)

        mu = Flatten(name='mu')(enc_layers[-1])

        latent_space = Lambda(compute_latent2, output_shape=K.int_shape(mu)[1:], name='latent')(mu)
        
        # Build the encoder
        self.encoder = Model(encoder_input, [mu, latent_space], name='encoder')

        self.encoder.summary()

        self.mu = mu; self.z = latent_space
        
        ''' DECODER '''
        # Take the convolution shape to be used in the decoder
        conv_shape = K.int_shape(enc_layers[-1])
        
        # Constructing decoder
        dec_inps = []
        decoder_input = Input(shape=K.int_shape(latent_space)[1:], name='dec_inp')
        dec_inps.append(decoder_input)
        
        if batch_norm2:
            dec2_1 = BatchNormalization()(decoder_input)
            dec3 = Reshape(conv_shape[1:])(dec2_1)
        else:
            dec3 = Reshape(conv_shape[1:])(decoder_input)


        '''
        dec_layers = [dec3]

        for i in range(self.nconv):
            dec_layers.append(Conv3D(hidden[::-1][i], (1,1,1), padding='valid', strides=(1,1,1), 
                                     activation=conv_act, name=f'dec_conv_dense_{i}')(dec_layers[-1]))
            dec_layers.append(BatchNormalization(name=f'dec_bn_dense_{i}')(dec_layers[-1]))

        dec_c1 = Conv3DTranspose(conv_filt, (2, 2, 1), padding='valid', strides=(2,2,1), 
                            activation=conv_act, name='dec_conv_1')(dec_layers[-1])
        dec_b1 = BatchNormalization(name='dec_batch_norm_0')(dec_c1)

        dec_c2 = Conv3DTranspose(conv_filt, (2, 2, 1), padding='valid', strides=(1,1,1), 
                            activation=conv_act, name='dec_conv_2')(dec_b1)
        dec_b2 = BatchNormalization(name='dec_batch_norm_1')(dec_c2)

        dec_c3 = Conv3DTranspose(conv_filt, (3, 3, 1), padding='valid', strides=(2,2,1), 
                            activation=conv_act, name='dec_conv_3')(dec_b2)
        dec_b3 = BatchNormalization(name='dec_batch_norm_2')(dec_c3)

        dec_c4 = Conv3DTranspose(32, (3, 3, 1), padding='valid', strides=(2,2,1), 
                            activation=conv_act, name='dec_conv_4')(dec_b3)
        dec_b4 = BatchNormalization(name='dec_batch_norm_3')(dec_c4)

        dec_c5 = Conv3DTranspose(4, (3, 3, 1), padding='valid', strides=(2,2,1), 
                            activation=conv_act, name='dec_conv_5')(dec_b4)
        dec_b5 = BatchNormalization(name='dec_batch_norm_4')(dec_c5)

        dec_c6 = Conv3DTranspose(1, (4, 4, 1), padding='valid', strides=(2,2,1), 
                            activation='relu', name='dec_conv_6')(dec_b5)
        '''

        dec_layers = decoder_layers(dec3, conv_filt, conv_act, hidden)
        #dec_b6 = BatchNormalization(name='dec_batch_norm_5')(dec_c6)

        #dec_c7 = Conv3D(1, (2, 2, 1), padding='valid', strides=(2,2,1), 
        #                    activation=conv_act, name='dec_conv_7')(dec_b6)

        decoder_output = Reshape(target_shape=input_shape, name='3d_reshape')(dec_layers[-1])
        # Build the decoder
        self.decoder = Model(decoder_input, decoder_output, name='decoder')
        self.decoder.summary()

        self.output = self.decoder(self.encoder(encoder_input)[1])
        
        # Build the VAE
        self.ae = Model(encoder_input, self.output, name='VAE')

        self.ae.encoder = self.encoder
        self.ae.decoder = self.decoder

        #self.ae.enc_conv_flat = Model(encoder_input, enc_inps[0], name='enc_conv_flat')
        #self.ae.enc_hidden    = Model(Input(tensor=enc_inps[0]), [mu, sigma, z], name='enc_hidden')

        # set the training parameters
        self.ae.sig0     = sigma0#*K.ones_like(sigma)
        self.ae.kl_beta  = beta
        
        self.ae.summary()
        
    def create_name(self):
        hidden_name = ''
        for layeri in self.hidden:
            hidden_name += '_%d'%layeri
        self.name = 'colorvae2_%dls_%dconv%d%s'%(self.latent_dim, self.nconv, self.conv_filt, hidden_name)

        if self.batch_norm:
            self.name += "_batchnorm"
        if self.batch_norm2:
            self.name += "_batchnorm2"

        if self.pool:
            self.name += "_pool"

        print(self.name)
    
    def add_loss_funcs(self):
        r_loss   = self.get_recon_loss()
        kl_loss  = self.get_kl_loss()

        # sum of all three losses
        loss = r_loss + kl_loss# + c_loss

        self.ae.add_loss(loss)
        self.ae.add_metric(r_loss, aggregation='mean', name='mse')
        self.ae.add_metric(kl_loss, aggregation='mean', name='kl')

class ConvVAE(BaseVariationalAE):
    def create_model(self, pool=False):
        hidden     = self.hidden
        conv_filt  = self.conv_filt
        conv_act   = self.conv_act

        input_conv = self.create_encoder()
        
        self.latent_dim = K.int_shape(self.mu)[-1]
        self.npixels    = K.int_shape(input_conv)[1]*K.int_shape(input_conv)[2]

        self.create_decoder(input_conv)

        self.output = self.decoder(self.encoder(self.input)[2])
        
        # Build the VAE
        self.ae = Model(self.input, self.output, name='VAE')

        self.ae.encoder = self.encoder
        self.ae.decoder = self.decoder

        self.ae.summary()
        
    def create_encoder(self):
        ''' ENCODER '''
        input_shape = (256, 256, 3)

        # Constructing encoder
        self.input = encoder_input = Input(shape=input_shape, name='input')

        #reshape_layer1 = Reshape(target_shape=(*input_shape, 1), name='3d_reshape')(self.input)
        
        enc_layers = []

        # convolution part
        enc_layers = encoder_layers2D_256(self.input, self.conv_filt, self.conv_act, self.hidden)
        input_conv = enc_layers[-1]
        
        mu = Flatten(name='mu')(enc_layers[-1])
        #sigma = Dense(K.int_shape(mu)[1], name='sig', activation=None)(Flatten()(enc_layers[-1]))
        sigma_conv = Conv2D(K.int_shape(enc_layers[-1])[-1], (1,1), name='sigma_conv')(enc_layers[-1])
        sigma      = Flatten(name='sigma')(sigma_conv)
        #sigma = tf.ones_like(mu, name='sigma')*sigma0

        latent_space = Lambda(compute_latent, output_shape=K.int_shape(mu)[1:], name='latent')([mu, sigma])
        
        # Build the encoder
        self.encoder = Model(encoder_input, [mu, sigma, latent_space], name='encoder')

        self.encoder.summary()

        self.mu = mu; self.sigma = sigma; self.z = latent_space
 
        return input_conv

    def create_decoder(self, input_conv):
        ''' DECODER '''
        # Take the convolution shape to be used in the decoder
        conv_shape = K.int_shape(input_conv)
        
        # Constructing decoder
        dec_inps = []
        decoder_input = Input(shape=K.int_shape(self.z)[1:], name='dec_inp')
        dec_inps.append(decoder_input)
        
        dec3 = Reshape(conv_shape[1:])(decoder_input)

        #upsamp_layer = UpSampling3D((2,2,1), name='dec_upsamp')(dec3)
        
        dec_layers = decoder_layers2D_256(dec3, self.conv_filt, self.conv_act, self.hidden)
        decoder_output = dec_layers[-1]
        # Build the decoder
        self.decoder = Model(decoder_input, decoder_output, name='decoder')
        self.decoder.summary()


    def create_name(self):
        hidden_name = ''
        for layeri in self.hidden:
            hidden_name += '_%d'%layeri
        self.name = 'convvae_%dls_%dconv%d%s'%(self.latent_dim, self.nconv, self.conv_filt, hidden_name)

        print(self.name)

class VAE_DEC(BaseVariationalAE):
    def create_model(self, sigma0=0., beta=1.e-3, conv_act='tanh', pool=False):
        global theta_p, u_p, lambda_p, n_centroid, latent_dim
        n_centroid = self.n_centroid
        hidden     = self.hidden
        conv_filt  = self.conv_filt
        batch_norm = self.batch_norm
        batch_norm2 = self.batch_norm2
        self.conv_act  = conv_act
        self.pool   = pool

        ''' ENCODER '''
        input_shape = (28, 28, 1)

        # Constructing encoder
        self.input = encoder_input = Input(shape=input_shape, name='input')

        flat = Flatten()(self.input)#Reshape(target_shape=(*input_shape, 1), name='hape')(self.input)
        
        # convolution part
        dense_layers = []
        dense_layers.append(Dense(500, activation=conv_act)(flat))
        dense_layers.append(Dense(500, activation=conv_act)(dense_layers[-1]))
        dense_layers.append(Dense(2000, activation=conv_act)(dense_layers[-1]))
        #dense_layers.append(Dense(16, activation=conv_act)(dense_layers[-1]))

        mu = Dense(10, name='mu', activation=None)(dense_layers[-1])
        latent_dim = mu.get_shape()[1]
        self.latent_dim = mu.get_shape()[1]
        #sigma = Dense(K.int_shape(mu)[1], name='sig', activation=None)(Flatten()(enc_layers[-1]))
        sigma = Dense(10, name='sigma', activation=None)(dense_layers[-1])

        latent_space = Lambda(compute_latent, output_shape=K.int_shape(mu)[1:], name='latent')([mu, sigma])
        
        # Build the encoder
        self.encoder = Model(encoder_input, [mu, sigma, latent_space], name='encoder')

        self.encoder.summary()

        self.mu = mu; self.sigma = sigma; self.z = latent_space
        
        self.gamma = GammaLayer(latent_dim, n_centroid, name='gamma')([mu, sigma, latent_space])
        
        ''' DECODER '''
        # Constructing decoder
        dec_inps = []
        decoder_input = Input(shape=K.int_shape(latent_space)[1:], name='dec_inp')
        dec_inps.append(decoder_input)

        dense_layers = []
        dense_layers.append(Dense(2000, activation=conv_act)(decoder_input))
        dense_layers.append(Dense(500, activation=conv_act)(dense_layers[-1]))
        dense_layers.append(Dense(500, activation=conv_act)(dense_layers[-1]))
        #dense_layers.append(Dense(1024, activation=conv_act)(dense_layers[-1]))
        dense_layers.append(Dense(K.int_shape(flat)[1], activation='relu')(dense_layers[-1]))
        
        decoder_output = unflat = Reshape(target_shape=input_shape)(dense_layers[-1])

        # Build the decoder
        self.decoder = Model(decoder_input, decoder_output, name='decoder')
        self.decoder.summary()

        self.output = self.decoder(self.encoder(encoder_input)[2])
        
        # Build the VAE
        self.ae = Model(encoder_input, [self.output, self.gamma], name='VAE')


        self.cluster = Model(encoder_input, self.gamma, name='DC')

        print(self.cluster.summary())

        self.ae.encoder = self.encoder
        self.ae.decoder = self.decoder

        #self.ae.enc_conv_flat = Model(encoder_input, enc_inps[0], name='enc_conv_flat')
        #self.ae.enc_hidden    = Model(Input(tensor=enc_inps[0]), [mu, sigma, z], name='enc_hidden')

        # set the training parameters
        self.ae.sig0     = sigma0#*K.ones_like(sigma)
        self.ae.kl_beta  = beta

        
        self.ae.summary()
        
    def add_loss_funcs(self):
        recon_loss = K.sum(K.square(self.input - self.output), axis=(1,2))#*128*128

        mui,sigi, z  = self.encoder(self.input)
        mup, sigp, zp = self.encoder(self.output)
        
        gamma_layer = self.ae.get_layer('gamma')

        gamma = self.cluster(self.input)
        Z, ZMean, ZLogVar = gamma_layer.get_z_vals([mui, sigi, z])
        gamma_t = K.repeat(gamma, self.latent_dim)

        batch_size = tf.shape(z)[0]

        theta_tensor3, u_tensor3, lambda_tensor3 = gamma_layer.get_tensors(batch_size)

        sig0 = self.ae.sig0*K.ones_like(mui)
        
        kl = - 1 + (K.square(mui-mup) + K.exp(sig0))/K.exp(sig0)
        kl = K.mean(kl, axis=-1)
        kl *= 0.5*self.ae.kl_beta

        r_loss   = tf.nn.compute_average_loss(recon_loss)
        kl_loss  = tf.nn.compute_average_loss(kl)    

        print(f"gamma_t: {gamma_t.get_shape()}, gamma: {gamma.get_shape()}")


        a = 0.5*K.sum(gamma*K.sum(K.log(lambda_tensor3)+
                        K.exp(ZLogVar)/lambda_tensor3+
                        K.square(ZMean-u_tensor3)/lambda_tensor3, axis=1),axis=(1))
        b = 0.5*K.sum(sigi+1,axis=-1)
        c = K.sum(K.log(theta_tensor3)*gamma_t, axis=(1,2))
        d = K.sum(K.log(K.mean(theta_tensor3,axis=1)/gamma)*gamma, axis=(1))

        print(a.get_shape(), b.get_shape(), c.get_shape(), d.get_shape())

        #print(gamma_t.get_shape(), lambda_tensor3.get_shape(), ZLogVar.get_shape(), lambda_tensor3.get_shape(),
        #      ZMean.get_shape(), u_tensor3.get_shape())
        clust_loss = tf.reduce_mean(a-b-d)
        # sum of all three losses
        loss = r_loss + clust_loss

        self.ae.add_loss(loss)
        self.ae.add_metric(r_loss, aggregation='mean', name='mse')
        self.ae.add_metric(kl_loss, aggregation='mean', name='kl')
        self.ae.add_metric(clust_loss, aggregation='mean', name='clust')
        self.ae.add_metric(K.min(lambda_tensor3), aggregation='mean', name='lambda_min')
        self.ae.add_metric(K.max(lambda_tensor3), aggregation='mean', name='lambda_max')
        self.ae.add_metric(K.min(theta_tensor3), aggregation='mean', name='theta_min')
        self.ae.add_metric(K.max(theta_tensor3), aggregation='mean', name='theta_max')
        self.ae.add_metric(K.min(u_tensor3), aggregation='mean', name='u_min')
        self.ae.add_metric(K.max(u_tensor3), aggregation='mean', name='u_max')
        self.ae.add_metric(K.min(gamma), aggregation='mean', name='gamma_min')
        self.ae.add_metric(K.max(gamma), aggregation='mean', name='gamma_max')

    def train_cluster(self, data, epochs=300, batch_size=10):
        savesfolder = self.get_savefolder()
        print(f"Training {self.name}")

        self.nepochs = epochs

        if not os.path.exists(savesfolder):
            os.mkdir(savesfolder)

        if hasattr(self, 'lr_scheduler'):
            print("Using Learning Rate Scheduler")
            self.history_clust = self.cluster.fit(data, epochs=epochs, validation_split=0.1,
                            callbacks=[self.lr_scheduler], validation_freq=5, 
                            batch_size=batch_size, shuffle=True)
        else:
            self.history_clust = self.cluster.fit(data, epochs=epochs, validation_split=0.1, 
                          validation_freq=5, batch_size=batch_size, shuffle=True)

    def set_initial_positions(self, data):
        from sklearn.mixture import GaussianMixture

        mu, sig, z = self.encoder.predict(data)
    
        gmm = GaussianMixture(self.n_centroid, covariance_type='diag')
        gmm.fit(z)

        gammalayer = self.ae.get_layer('gamma')
        gammalayer.u_p[:].assign(gmm.means_.reshape(1, self.latent_dim, self.n_centroid))
        gammalayer.lambda_p[:].assign(gmm.covariances_.reshape(1, self.latent_dim, self.n_centroid)**2.)

    def create_name(self):
        hidden_name = ''
        for layeri in self.hidden:
            hidden_name += '_%d'%layeri
        self.name = 'mnistvae_cluster_%dls_%dconv%d%s'%(self.latent_dim, self.nconv, self.conv_filt, hidden_name)

        if self.batch_norm:
            self.name += "_batchnorm"
        if self.batch_norm2:
            self.name += "_batchnorm2"

        if self.pool:
            self.name += "_pool"

        print(self.name)
    
    def save(self):
        savesfolder = self.get_savefolder()
        self.ae.encoder.save_weights(savesfolder+"encoderw.h5")
        self.ae.decoder.save_weights(savesfolder+"decoderw.h5")
        self.cluster.save_weights(savesfolder+"clusterw.h5")
        self.ae.save_weights(savesfolder+"VAEw.h5")
    
    def load(self):
        savesfolder = self.get_savefolder()
        self.ae.encoder.load_weights(savesfolder+"encoderw.h5")
        self.ae.decoder.load_weights(savesfolder+"decoderw.h5")
        self.cluster.load_weights(savesfolder+"clusterw.h5")
        #self.ae.load_weights(savesfolder+"VAEw.h5")
    
    def load_vae_weights(self, savesfolder):
        self.ae.encoder.load_weights(savesfolder+"encoderw.h5")
        self.ae.decoder.load_weights(savesfolder+"decoderw.h5")

class VAE_mnist(BaseVariationalAE):
    def create_model(self, sigma0=0., beta=1.e-3, conv_act='tanh', pool=False):
        global theta_p, u_p, lambda_p, n_centroid, latent_dim
        n_centroid = self.n_centroid
        hidden     = self.hidden
        conv_filt  = self.conv_filt
        batch_norm = self.batch_norm
        batch_norm2 = self.batch_norm2
        self.conv_act  = conv_act
        self.pool   = pool

        ''' ENCODER '''
        input_shape = (28, 28, 1)

        # Constructing encoder
        self.input = encoder_input = Input(shape=input_shape, name='input')

        flat = Flatten()(self.input)#Reshape(target_shape=(*input_shape, 1), name='hape')(self.input)
        
        # convolution part
        dense_layers = []
        dense_layers.append(Dense(500, activation=conv_act)(flat))
        dense_layers.append(Dense(500, activation=conv_act)(dense_layers[-1]))
        dense_layers.append(Dense(2000, activation=conv_act)(dense_layers[-1]))
        #dense_layers.append(Dense(16, activation=conv_act)(dense_layers[-1]))

        mu = Dense(10, name='mu', activation=None)(dense_layers[-1])
        latent_dim = mu.get_shape()[1]
        self.latent_dim = mu.get_shape()[1]
        #sigma = Dense(K.int_shape(mu)[1], name='sig', activation=None)(Flatten()(enc_layers[-1]))
        sigma = Dense(10, name='sigma', activation=None)(dense_layers[-1])

        latent_space = Lambda(compute_latent, output_shape=K.int_shape(mu)[1:], name='latent')([mu, sigma])
        
        # Build the encoder
        self.encoder = Model(encoder_input, [mu, sigma, latent_space], name='encoder')

        self.encoder.summary()

        self.mu = mu; self.sigma = sigma; self.z = latent_space
        
        ''' DECODER '''
        # Constructing decoder
        dec_inps = []
        decoder_input = Input(shape=K.int_shape(latent_space)[1:], name='dec_inp')
        dec_inps.append(decoder_input)

        dense_layers = []
        dense_layers.append(Dense(2000, activation=conv_act)(decoder_input))
        dense_layers.append(Dense(500, activation=conv_act)(dense_layers[-1]))
        dense_layers.append(Dense(500, activation=conv_act)(dense_layers[-1]))
        #dense_layers.append(Dense(1024, activation=conv_act)(dense_layers[-1]))
        dense_layers.append(Dense(K.int_shape(flat)[1], activation='relu')(dense_layers[-1]))
        
        decoder_output = unflat = Reshape(target_shape=input_shape)(dense_layers[-1])

        # Build the decoder
        self.decoder = Model(decoder_input, decoder_output, name='decoder')
        self.decoder.summary()

        self.output = self.decoder(self.encoder(encoder_input)[2])
        
        # Build the VAE
        self.ae = Model(encoder_input, self.output, name='VAE')

        self.ae.encoder = self.encoder
        self.ae.decoder = self.decoder

        #self.ae.enc_conv_flat = Model(encoder_input, enc_inps[0], name='enc_conv_flat')
        #self.ae.enc_hidden    = Model(Input(tensor=enc_inps[0]), [mu, sigma, z], name='enc_hidden')

        # set the training parameters
        self.ae.sig0     = sigma0#*K.ones_like(sigma)
        self.ae.kl_beta  = beta

        
        self.ae.summary()
        
    def add_loss_funcs(self):
        global theta_p, u_p, lambda_p

        recon_loss = K.sum(K.square(self.input - self.output), axis=(1,2))#*128*128

        mui,sigi, z  = self.encoder(self.input)
        mup, sigp, zp = self.encoder(self.output)
        
        sig0 = self.ae.sig0*K.ones_like(mui)
        
        kl = - 1 + (K.square(mui-mup) + K.exp(sig0))/K.exp(sig0)
        kl = K.mean(kl, axis=-1)
        kl *= 0.5*self.ae.kl_beta

        r_loss   = tf.nn.compute_average_loss(recon_loss)
        kl_loss  = tf.nn.compute_average_loss(kl)    

        loss = r_loss + kl_loss

        self.ae.add_loss(loss)
        self.ae.add_metric(r_loss, aggregation='mean', name='mse')
        self.ae.add_metric(kl_loss, aggregation='mean', name='kl')

    def create_name(self):
        hidden_name = ''
        for layeri in self.hidden:
            hidden_name += '_%d'%layeri
        self.name = 'mnistvae_cluster_%dls_%dconv%d%s'%(self.latent_dim, self.nconv, self.conv_filt, hidden_name)

        if self.batch_norm:
            self.name += "_batchnorm"
        if self.batch_norm2:
            self.name += "_batchnorm2"

        if self.pool:
            self.name += "_pool"

        print(self.name)
    
    def save(self):
        savesfolder = self.get_savefolder()
        self.ae.encoder.save_weights(savesfolder+"encoderw.h5")
        self.ae.decoder.save_weights(savesfolder+"decoderw.h5")
        self.ae.save_weights(savesfolder+"VAEw.h5")
    
    def load(self):
        savesfolder = self.get_savefolder()
        self.ae.encoder.load_weights(savesfolder+"encoderw.h5")
        self.ae.decoder.load_weights(savesfolder+"decoderw.h5")
        #self.ae.load_weights(savesfolder+"VAEw.h5")
