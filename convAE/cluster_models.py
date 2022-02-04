from .globals import *
from .base_models import BaseVariationalAE, GammaLayerConv, GammaLayer 
from .conv_models import ConvVAE

class ConvVAE_DEC(ConvVAE):
    def create_model(self, input_size=128):
        global theta_p, u_p, lambda_p, n_centroid, latent_dim
        n_centroid = self.n_centroid
        hidden     = self.hidden
        conv_filt  = self.conv_filt
        self.conv_act  = conv_act

        ''' ENCODER '''
        input_conv = self.create_encoder(input_size)

        self.latent_dim = K.int_shape(self.mu)[-1]
        self.npixels    = K.int_shape(input_conv)[1]*K.int_shape(input_conv)[2]

        ''' CLUSTERING LAYER '''
        self.gamma = GammaLayerConv(self.latent_dim, self.n_centroid, self.npixels, 
                                    name='gamma')([self.mu, self.sigma, self.z])
        
        ''' DECODER '''
        self.create_decoder(input_conv, input_size)

        self.output = self.decoder(self.encoder(self.input)[2])
        
        # Build the VAE
        self.ae = Model(self.input, [self.output, self.gamma], name='VAE')

        self.cluster = Model(self.input, self.gamma, name='DC')
        self.cluster.summary()

        self.ae.summary()
        
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
        self.name = 'convvae_cluster_%dls_%dconv%d%s'%(self.latent_dim, self.nconv, self.conv_filt, hidden_name)

        print(self.name)
    
    def load_vae_weights(self, savesfolder):
        self.ae.encoder.load_weights(savesfolder+"encoderw.h5")
        self.ae.decoder.load_weights(savesfolder+"decoderw.h5")
    
    def add_loss_funcs(self):
        r_loss   = self.get_recon_loss()
        kl_loss  = self.get_kl_loss()
        c_loss   = self.get_cluster_loss()

        # sum of all three losses
        loss = r_loss + kl_loss + c_loss

        self.ae.add_loss(loss)
        self.ae.add_metric(r_loss, aggregation='mean', name='mse')
        self.ae.add_metric(kl_loss, aggregation='mean', name='kl')
        self.ae.add_metric(c_loss, aggregation='mean', name='cluster')
