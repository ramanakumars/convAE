from .globals import *
from .base_models import * 
from .conv_models import *

class ConvVAE_DEC(ConvVAE):
    def create_model(self, input_size=128):
        global theta_p, u_p, lambda_p, n_centroid, latent_dim
        n_centroid = self.n_centroid
        hidden     = self.hidden
        conv_filt  = self.conv_filt

        ''' ENCODER '''
        input_conv = self.create_encoder(input_size)

        self.latent_dim = K.int_shape(self.mu)[-1]
        self.reduced_latent_dim = K.int_shape(input_conv)[3]
        self.npixels    = K.int_shape(input_conv)[1]*K.int_shape(input_conv)[2]

        ''' CLUSTERING LAYER '''
        self.gammalayer  = GammaLayerConv(self.latent_dim, self.n_centroid, self.npixels, 
                                    name='gamma')
        self.gamma = self.gammalayer([self.mu, self.sigma, self.z])
        
        ''' DECODER '''
        self.create_decoder(input_conv, input_size)

        self.output = self.decoder(self.encoder(self.input)[2])
        
        # Build the VAE
        self.ae = Model(self.input, [self.output, self.gamma], name='VAE')

        self.cluster = Model(self.input, self.gamma, name='DC')
        self.cluster.summary()
        
        self.ae.encoder = self.encoder
        self.ae.decoder = self.decoder

        self.ae.summary()
        
    def set_initial_positions(self, data):
        from sklearn.mixture import GaussianMixture

        mu, sig, z = self.encoder.predict(data)
        
        z = z.reshape((len(z), self.npixels, self.reduced_latent_dim))
        u   = np.zeros((1, self.npixels, self.reduced_latent_dim, self.n_centroid), dtype=np.float32)
        lam = np.zeros((1, self.npixels, self.reduced_latent_dim, self.n_centroid), dtype=np.float32)
        gammalayer  = self.ae.get_layer('gamma')

        for i in range(self.npixels):
            gmm = GaussianMixture(self.n_centroid, covariance_type='diag')
            gmm.fit(z[:,i,:])
            u[0,i,:] = gmm.means_.T
            lam[0,i,:] = gmm.covariances_.T
        gammalayer.u_p.assign(u)
        #gammalayer.lambda_p[:].assign(lam)
    
    def set_initial_positions_test(self):
        u   = np.zeros((1, self.npixels, self.reduced_latent_dim, self.n_centroid), dtype=np.float32)
        lam = np.zeros((1, self.npixels, self.reduced_latent_dim, self.n_centroid), dtype=np.float32)
        gammalayer  = self.ae.get_layer('gamma')
        gammalayer.u_p.assign(u)
            
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
        loss = r_loss + kl_loss + 0.1*c_loss

        self.ae.add_loss(loss)
        self.ae.add_metric(r_loss, aggregation='mean', name='mse')
        self.ae.add_metric(kl_loss, aggregation='mean', name='kl')
        self.ae.add_metric(c_loss, aggregation='mean', name='cluster')

class ConvAE_DEC(ConvAE):
    def create_model(self, input_size=128):
        global theta_p, u_p, lambda_p, n_centroid, latent_dim
        n_centroid = self.n_centroid
        hidden     = self.hidden
        conv_filt  = self.conv_filt

        ''' ENCODER '''
        input_conv = self.create_encoder(input_size)

        self.latent_dim = K.int_shape(self.mu)[-1]
        self.reduced_latent_dim = K.int_shape(input_conv)[3]
        self.npixels    = K.int_shape(input_conv)[1]*K.int_shape(input_conv)[2]

        ''' CLUSTERING LAYER '''
        self.gamma = ClusteringLayer(self.latent_dim, self.n_centroid, self.npixels, 
                                    name='cluster')(self.z)
        
        ''' DECODER '''
        self.create_decoder(input_conv, input_size)

        self.output = self.decoder(self.encoder(self.input))
        
        # Build the VAE
        self.ae = Model(self.input, [self.output, self.gamma], name='VAE')

        self.cluster = Model(self.input, self.gamma, name='DC')
        self.cluster.summary()
        
        self.ae.encoder = self.encoder
        self.ae.decoder = self.decoder

        self.ae.summary()
        
    def set_initial_positions(self, data):
        from sklearn.cluster import KMeans

        z = self.encoder.predict(data)
        kmeans = KMeans(n_clusters=self.n_centroid).fit(z)

        gammalayer = self.ae.get_layer('cluster')
        gammalayer.set_weights(kmeans.cluster_centers_.reshape(self.npixels, self.reduced_latent_dim, self.n_centroid))
    
    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def create_name(self):
        hidden_name = ''
        for layeri in self.hidden:
            hidden_name += '_%d'%layeri
        self.name = 'convae_cluster_%dls_%dconv%d%s'%(self.latent_dim, self.nconv, self.conv_filt, hidden_name)

        print(self.name)
    
    def load_vae_weights(self, savesfolder):
        self.ae.encoder.load_weights(savesfolder+"encoderw.h5")
        self.ae.decoder.load_weights(savesfolder+"decoderw.h5")

    def get_cluster_loss(self):
        q = self.cluster.predict(self.input)
        p = self.target_distribution(q)

        cluster_loss = tf.reduce_mean(K.mean(K.sum(KLDivergence()(p, q), axis=(3)), axis=(1,2)))
        return cluster_loss
    
    def add_loss_funcs(self):
        r_loss   = self.get_recon_loss()
        c_loss   = self.get_cluster_loss()

        # sum of all three losses
        loss = r_loss + 0.01*c_loss

        self.ae.add_loss(loss)
        self.ae.add_metric(r_loss, aggregation='mean', name='mse')
        self.ae.add_metric(c_loss, aggregation='mean', name='cluster')
