from .globals import *

class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.
    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, latent_dim, n_centroid, npixels, alpha=1.0, **kwargs):
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_centroid = n_centroid
        self.latent_dim = latent_dim
        self.npixels    = npixels
        self.alpha      = 1.0
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, self.latent_dim*self.npixels))

    def build(self, input_shape):
        self.clusters = self.add_weight(shape=(self.npixels, self.latent_dim, self.n_centroid), initializer='glorot_uniform', name='clusters')
        self.built = True
    
    def get_z_vals(self, x, only_z=False):
        n_centroid = self.n_centroid; latent_dim = self.latent_dim; npixels = self.npixels
        
        # reshape the latent values
        Z = tf.transpose(K.repeat(x, n_centroid),perm=[0,2,1])
        Z = Reshape((self.npixels, self.latent_dim, n_centroid))(Z)

        return Z

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        Z = self.get_z_vals(inputs)
        q = 1.0 / (1.0 + (K.sum(K.square(Z - self.clusters), axis=3) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=(1,2)))
        return q

    def compute_output_shape(self, input_shape):
        return (self.latent_dim, self.npixels, self.n_clusters)

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

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
        self.lambda_p = np.exp(-2)*tf.ones(shape=(1, self.npixels, self.latent_dim, self.n_centroid), dtype=tf.float32)
        #self.lambda_p = tf.Variable(lambda_init, trainable=True, shape=(1,self.npixels, self.latent_dim, self.n_centroid),name="lambda", dtype=tf.float32)

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
