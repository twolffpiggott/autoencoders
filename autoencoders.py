from keras.layers import Input, Dense, Lambda
from keras.models import Model

from custom import Sampler, CustomVariationalLayer

import plotting

class VariationalAutoencoder:
    def __init__(self, **kwargs):
        self.batch_size = kwargs.get('batch_size')
        self.original_dim = kwargs.get('original_dim')
        self.latent_dim = kwargs.get('latent_dim')
        self.intermediate_dim = kwargs.get('intermediate_dim')
        self.epsilon_std = kwargs.get('epsilon_std')
    
        x = Input(batch_shape=(self.batch_size, self.original_dim))
        h = Dense(self.intermediate_dim, activation='relu')(x)
        z_mean = Dense(self.latent_dim)(h)
        z_log_var = Dense(self.latent_dim)(h)

        sampler = Sampler(batch_size=self.batch_size, latent_dim=self.latent_dim, epsilon_std=self.epsilon_std)
        z = Lambda(sampler.sampling, output_shape=(self.latent_dim,))([z_mean, z_log_var])

        decoder_h = Dense(self.intermediate_dim, activation='relu')
        decoder_mean = Dense(self.original_dim, activation='sigmoid')
        h_decoded = decoder_h(z)
        x_decoded_mean = decoder_mean(h_decoded)

        y = CustomVariationalLayer(self.original_dim, z_mean, z_log_var)([x, x_decoded_mean])
        self.vae = Model(x, y)
        self.vae.compile(optimizer='rmsprop', loss=None)
        
        self.encoder = Model(x, z_mean)

        decoder_input = Input(shape=(self.latent_dim,))
        _h_decoded = decoder_h(decoder_input)
        _x_decoded_mean = decoder_mean(_h_decoded)
        self.generator = Model(decoder_input, _x_decoded_mean)

    def train(self, x_train, x_test, epochs):
        self.vae.fit(x_train,
                shuffle=True,
                epochs=epochs,
                batch_size=self.batch_size,
                validation_data=(x_test, x_test))
    
    def encode(self, x_test):
        x_test_encoded = self.encoder.predict(x_test, batch_size=self.batch_size)
        return x_test_encoded

    def generate(self, z_sample):
        x_decoded = self.generator.predict(z_sample)
        return x_decoded

    def report(self, x_test, y_test):
        plotting.plot_scatter(self.encode(x_test), y_test)
        plotting.plot_manifold(self.generator)
        
