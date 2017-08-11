from keras import backend as K
from keras.layers import Layer
from keras import metrics

class Sampler:
    def __init__(self, **kwargs):
        self.batch_size = kwargs.get('batch_size')
        self.latent_dim = kwargs.get('latent_dim')
        self.epsilon_std = kwargs.get('epsilon_std')
    
    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(self.batch_size, self.latent_dim), mean=0., stddev=self.epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon

class CustomVariationalLayer(Layer):
    def __init__(self, original_dim, z_mean, z_log_var, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)
        self.original_dim = original_dim
        self.z_mean = z_mean
        self.z_log_var = z_log_var

    def vae_loss(self, x, x_decoded_mean):
        xent_loss = self.original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)
    
    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean = inputs[1]
        loss = self.vae_loss(x, x_decoded_mean)
        self.add_loss(loss, inputs=inputs)
        return x
