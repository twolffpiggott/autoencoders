import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import helpers

def plot_scatter(encoded, labels):
    plt.clf(); plt.cla()
    plt.figure()
    plt.scatter(encoded[:,0], encoded[:,1], c=labels, cmap='rainbow')
    plt.colorbar()
    helpers.multivac_save_graph('latent_space_scatter_2d')

def plot_manifold(generator, n=15, digit_size=28):
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly-spaced coordinates on the unit square are transformed through the inverse CDF (ppf) of the Gaussian
    # to produce values of the latent variables z, since the prior of the latent space is Gaussian
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            x_decoded = generator.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size] = digit
    plt.clf(); plt.cla()
    plt.figure()
    plt.imshow(figure, cmap='Greys_r')
    helpers.multivac_save_graph('manifold_plot_2d')
