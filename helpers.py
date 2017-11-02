# mnist read imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import tempfile

import numpy 
from six.moves import urllib
from six.moves import xrange # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

# general utility imports
import multivac as m
import pandas as pd
import numpy as np
from datetime import datetime
import re

from keras.datasets import mnist

proj = 'autoencoders'

def multivac_persist(data, dname):
    m.persist.core.write_data(proj, dname, df, verbose=False)

def multivac_get(dname):
    d = m.get_data(proj, dname, 'latest', load_as_str=False, verbose=False)
    return d

def multivac_save_graph(gname):
    m.persist.graph.save_graph(proj, title=gname, date=datetime.now(), verbose=False)

def prepare_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    return x_train, y_train, x_test, y_test
    
def get_mnist():
    return read_data_sets('MNIST_data', one_hot=True)
