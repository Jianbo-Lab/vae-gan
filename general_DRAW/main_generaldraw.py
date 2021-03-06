
# coding: utf-8

# In[185]:

from __future__ import division, print_function, absolute_import
import numpy as np


import sys
sys.path.append('../core/')
sys.path.append('../networks/')
from general_draw import generalDRAW
from networks import *

import numpy as np
import tensorflow as tf
import argparse

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('condition',type = bool)
    parser.add_argument('gan',type = bool)
    parser.add_argument('enc_size',type = int)
    parser.add_argument('dec_size',type = int) 
    parser.add_argument('learning_rate',type = float) 
    parser.add_argument('train_itrs',type = int)
    args = parser.parse_args()

    condition = args.condition
    gan = args.gan
    lr = args.learning_rate
    train_itrs = args.train_itrs
    enc_size = args.enc_size
    dec_size = args.dec_size

    tf.reset_default_graph()
    
    sess = tf.InteractiveSession()
    build_encoder = Encoder(enc_size)
    build_decoder = Decoder(dec_size)
    build_discriminator = Discriminator_v3(100)
    print_itrs = 100


    model =  generalDRAW(sess, build_encoder, build_decoder, build_discriminator,
                  read_attn = True, write_attn = True, T = 10,
                  batch_size = 256, enc_size = enc_size,
                         dec_size = dec_size, condition = condition, gan = gan)

    model.train(train_itrs = train_itrs, learning_rate = lr, print_itrs = print_itrs)
    


if __name__ == '__main__':
    main()

