# autoencoders

## Usage for Keras Variational Autoencoder
```python
import autoencoders
import helpers
# read in MNIST data
x_train, y_train, x_test, y_test = helpers.prepare_data()
# initialize and compile variational autoencoder
vae = autoencoders.VariationalAutoencoder(batch_size=100, original_dim=784, latent_dim=2, intermediate_dim=256, epsilon_std=1.0)
# train the model
vae.train(x_train, x_test, 10)
# plot encoded scatterplot and manifold visualization
vae.report(x_test, y_test)
```

## Usage for Tensorflow Variational Autoencoder
### slight modification of [jmetzen's](https://jmetzen.github.io/2015-11-27/vae.html "jmetzen's walkthrough") awesome implementation
``` python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import helpers
from autoencoders import TFVariationalAutoencoder

# grab mnist data
mnist = helpers.read_data_sets('MNIST_data', one_hot=True)

network_architecture = \
    dict(n_hidden_recog_1=500, # 1st layer encoder neurons
         n_hidden_recog_2=500, # 2nd layer encoder neurons
         n_hidden_gener_1=500, # 1st layer decoder neurons
         n_hidden_gener_2=500, # 2nd layer decoder neurons
         n_input=784, # MNIST data input (img shape: 28*28)
         n_z=2)  # dimensionality of latent space

# initialize and train tensorflow VAE
vae = TFVariationalAutoencoder(network_architecture, 
                             learning_rate=0.001, 
                             batch_size=100)
vae = vae.train(mnist=mnist,
                training_epochs=1,
                display_step=5)

# plot the resulting latent space representation
x_sample, y_sample = mnist.test.next_batch(5000)
z_mu = vae.transform(x_sample)
plt.figure(figsize=(8, 6)) 
plt.scatter(z_mu[:, 0], z_mu[:, 1], c=np.argmax(y_sample, 1), cmap='rainbow')
plt.colorbar()
plt.grid()
plt.show()

```

