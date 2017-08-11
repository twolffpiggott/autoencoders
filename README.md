# autoencoders
## Usage for Keras Variational autoencoder

{% highlight python %}
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
{% endhighlight %}


