from .globals import *
from .model import VariationalAE

class ConvAE(VariationalAE):
    def create_model(self, conv_act='tanh', pool=False):
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

        #reshape_layer1 = Reshape(target_shape=(*input_shape, 1), name='3d_reshape')(self.input)
        
        enc_layers = []

        # convolution part
        enc_layers = encoder_layers2D(self.input, conv_filt, conv_act, hidden)
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
        
        if batch_norm2:
            dec2_1 = BatchNormalization()(decoder_input)
            dec3 = Reshape(conv_shape[1:])(dec2_1)
        else:
            dec3 = Reshape(conv_shape[1:])(decoder_input)

        #upsamp_layer = UpSampling3D((2,2,1), name='dec_upsamp')(dec3)
        
        dec_layers = decoder_layers2D(dec3, conv_filt, conv_act, hidden)
        decoder_output = dec_layers[-1]
        # Build the decoder
        self.decoder = Model(decoder_input, decoder_output, name='decoder')
        self.decoder.summary()

        self.output = self.decoder(self.encoder(encoder_input))
        
        # Build the VAE
        self.ae = Model(encoder_input, self.output, name='VAE')

        self.ae.encoder = self.encoder
        self.ae.decoder = self.decoder
        
        # set the training parameters
        self.ae.summary()
        
    def create_name(self):
        hidden_name = ''
        for layeri in self.hidden:
            hidden_name += '_%d'%layeri
        self.name = 'convae_%dls_%dconv%d%s'%(self.latent_dim, self.nconv, self.conv_filt, hidden_name)

        if self.batch_norm:
            self.name += "_batchnorm"
        if self.batch_norm2:
            self.name += "_batchnorm2"

        if self.pool:
            self.name += "_pool"

        print(self.name)
    
    def add_loss_funcs(self):
        recon_loss = K.mean(K.sum(K.abs(self.input - self.output), axis=(1,2)), axis=(1))#*128*128

        z = self.encoder(self.input)
        zp = self.encoder(self.output)

        z_mse_loss = K.sum(K.square(z - zp), axis=1)

        r_loss   = tf.nn.compute_average_loss(recon_loss)
        z_loss   = tf.nn.compute_average_loss(z_mse_loss)

        # sum of all three losses
        loss = r_loss + z_loss# + c_loss

        self.ae.add_loss(loss)
        self.ae.add_metric(r_loss, aggregation='mean', name='mse')
        self.ae.add_metric(z_loss, aggregation='mean', name='z')

