import multivac as m
import pandas as pd
import numpy as np
from datetime import datetime
import re

from keras.datasets import mnist

proj = 'autoencoders'

def multivac_persist(data, dname):
    m.persist.core.write_data(proj, dname, datetime.now(), data, verbose=False)
    dconf = m.persist.core.autogenerate_dconf(proj, dname, 'latest')
    m.persist.core.write_dconf(proj, dname, dconf, verbose=False)

def multivac_get(dname):
    d, _ = m.persist.core.get_data(proj, dname, 'latest', None, verbose=False)
    return d

def multivac_save_graph(gname):
    m.persist.review.save_graph(proj=proj, gname=gname, date=datetime.now(), verbose=False)

def prepare_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    return x_train, y_train, x_test, y_test
