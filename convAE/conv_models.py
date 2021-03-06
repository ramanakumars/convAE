from .globals import *
from .base_models import *

class ConvVAE(BaseVariationalAE):
    def create_model(self, input_size=128):
        hidden     = self.hidden
        conv_filt  = self.conv_filt
        conv_act   = self.conv_act

        input_conv = self.create_encoder(input_size)
        
        self.latent_dim = K.int_shape(self.mu)[-1]
        self.npixels    = K.int_shape(input_conv)[1]*K.int_shape(input_conv)[2]

        self.create_decoder(input_conv, input_size)

        self.output = self.decoder(self.encoder(self.input)[2])
        
        # Build the VAE
        self.ae = Model(self.input, self.output, name='VAE')

        self.ae.encoder = self.encoder
        self.ae.decoder = self.decoder

        self.ae.summary()
        
    def create_encoder(self, input_size):
        ''' ENCODER '''

        if input_size==128:
            input_shape = (128, 128, 3)
        else:
            input_shape = (256, 256, 3)

        # Constructing encoder
        self.input = encoder_input = Input(shape=input_shape, name='input')

        #reshape_layer1 = Reshape(target_shape=(*input_shape, 1), name='3d_reshape')(self.input)
        
        enc_layers = []

        # convolution part
        if input_size==128:
            enc_layers = encoder_layers2D(self.input, self.conv_filt, self.conv_act, self.hidden)
        else:
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

    def create_decoder(self, input_conv, input_size):
        ''' DECODER '''
        # Take the convolution shape to be used in the decoder
        conv_shape = K.int_shape(input_conv)
        
        # Constructing decoder
        dec_inps = []
        decoder_input = Input(shape=K.int_shape(self.z)[1:], name='dec_inp')
        dec_inps.append(decoder_input)
        
        dec3 = Reshape(conv_shape[1:])(decoder_input)

        #upsamp_layer = UpSampling3D((2,2,1), name='dec_upsamp')(dec3)
        
        if input_size==128:
            dec_layers = decoder_layers2D(dec3, self.conv_filt, self.conv_act, self.hidden)
        else:
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


class ConvAE(BaseVariationalAE):
    def create_model(self, input_size=128):
        hidden     = self.hidden
        conv_filt  = self.conv_filt
        conv_act   = self.conv_act

        input_conv = self.create_encoder(input_size)
        
        self.latent_dim = K.int_shape(self.z)[-1]
        self.npixels    = K.int_shape(input_conv)[1]*K.int_shape(input_conv)[2]

        self.create_decoder(input_conv, input_size)

        self.output = self.decoder(self.encoder(self.input))
        
        # Build the VAE
        self.ae = Model(self.input, self.output, name='AE')

        self.ae.encoder = self.encoder
        self.ae.decoder = self.decoder

        self.ae.summary()
        
    def create_encoder(self, input_size):
        ''' ENCODER '''

        if input_size==128:
            input_shape = (128, 128, 3)
        else:
            input_shape = (256, 256, 3)

        # Constructing encoder
        self.input = encoder_input = Input(shape=input_shape, name='input')

        #reshape_layer1 = Reshape(target_shape=(*input_shape, 1), name='3d_reshape')(self.input)
        
        enc_layers = []

        # convolution part
        if input_size==128:
            enc_layers = encoder_layers2D(self.input, self.conv_filt, self.conv_act, self.hidden)
        else:
            enc_layers = encoder_layers2D_256(self.input, self.conv_filt, self.conv_act, self.hidden)

        input_conv = enc_layers[-1]
        
        latent_space = Flatten(name='latent')(enc_layers[-1])
        
        # Build the encoder
        self.encoder = Model(encoder_input, latent_space, name='encoder')

        self.encoder.summary()

        self.z = latent_space
 
        return input_conv

    def create_decoder(self, input_conv, input_size):
        ''' DECODER '''
        # Take the convolution shape to be used in the decoder
        conv_shape = K.int_shape(input_conv)
        
        # Constructing decoder
        dec_inps = []
        decoder_input = Input(shape=K.int_shape(self.z)[1:], name='dec_inp')
        dec_inps.append(decoder_input)
        
        dec3 = Reshape(conv_shape[1:])(decoder_input)

        #upsamp_layer = UpSampling3D((2,2,1), name='dec_upsamp')(dec3)
        
        if input_size==128:
            dec_layers = decoder_layers2D(dec3, self.conv_filt, self.conv_act, self.hidden)
        else:
            dec_layers = decoder_layers2D_256(dec3, self.conv_filt, self.conv_act, self.hidden)

        decoder_output = dec_layers[-1]
        # Build the decoder
        self.decoder = Model(decoder_input, decoder_output, name='decoder')
        self.decoder.summary()


    def create_name(self):
        hidden_name = ''
        for layeri in self.hidden:
            hidden_name += '_%d'%layeri
        self.name = 'convae_%dls_%dconv%d%s'%(self.latent_dim, self.nconv, self.conv_filt, hidden_name)

        print(self.name)

    def add_loss_funcs(self):
        r_loss   = self.get_recon_loss()

        # sum of all three losses
        loss = r_loss# + kl_loss# + c_loss

        self.ae.add_loss(loss)
        self.ae.add_metric(r_loss, aggregation='mean', name='mse')
        #self.ae.add_metric(kl_loss, aggregation='mean', name='kl')
