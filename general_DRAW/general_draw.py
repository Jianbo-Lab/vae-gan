# This generalDRAW is based on the TensorFlow implementation of vanilla DRAW by https://github.com/ericjang/draw
# Usage:
# Use the main_generalDRAW.py in command line:
# python main_generaldraw.py True True 256 256 0.001 1000
# The input parameters are: encoder hidden dimension, decoder hidden dimension, learning rate, and learning iterations



import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials import mnist
import time
import os
from helper import *



class generalDRAW():

    def __init__(self, sess, build_encoder, build_decoder, build_discriminator, read_attn = False, write_attn = False, T = 10, dec_size = 128, enc_size = 128, batch_size = 256,condition = False, gan = False, z_size = 10):
        self.sess = sess

        self.gan = gan
        self.condition = condition
        self.build_encoder = build_encoder
        self.build_decoder = build_decoder
        self.build_discriminator = build_discriminator
        self.batch_size  = batch_size
    
        self.dec_size = dec_size
        self.enc_size = enc_size

        self.DO_SHARE = None 
        self.data_dir = ""
        self.read_attn = read_attn
        self.write_attn = write_attn


        # Currently only for MNIST data, which image is represented as 28 by 28 matrix
        self.A, self.B = 28, 28 
        self.x_width = 28
        self.img_size = self.B * self.A # the canvas size
        # self.enc_size = 256 # number of hidden units / output size in LSTM
        # self.dec_size = 256
        self.read_n = 5 # read glimpse grid width/height
        self.write_n = 5 # write glimpse grid width/height
        self.read_size = 2* self.read_n * self.read_n if self.read_attn else 2* self.img_size
        self.write_size = self.write_n * self.write_n if self.write_attn else self.img_size


        self.z_size = z_size # QSampler output size
        self.T = T # MNIST generation sequence length
        self.eps=1e-8 # epsilon for numerical stability

    def train(self, train_itrs = 1000, print_itrs = 1000, learning_rate = 1e-4, load_file = None):

        # self.batch_size = batch_size # training minibatch size
        self.train_itrs = train_itrs
        self.print_itrs = print_itrs
        self.learning_rate= learning_rate # learning rate for optimizer

        #---------------------------------------------------------------
        #---------------------------------------------------------------
        #---------------------------------------------------------------
        # Build Graph

        enc_loss, dec_loss, draw_loss, dis_loss = self.build_DRAW()

        if not self.gan:
            # If do not add gan, only compute encoder loss + decoder loss
            draw_loss = enc_loss +  dec_loss


        ## OPTIMIZER ## 
        
        # optimizer=tf.train.AdamOptimizer(self.learning_rate, beta1 = 0.5)
        # grads=optimizer.compute_gradients(cost)

        # for i,(g,v) in enumerate(grads):
        #     if g is not None:
        #         grads[i]=(tf.clip_by_norm(g,5),v) # clip gradients
        # train_op = optimizer.apply_gradients(grads)
                
        ## RUN TRAINING ## 

        data_directory = os.path.join(self.data_dir, "mnist")
        if not os.path.exists(data_directory):
	    os.makedirs(data_directory)

        # binarized (0-1) mnist data

        train_data = mnist.input_data.read_data_sets(data_directory, one_hot=True).train 


        
        sess = self.sess
        
        
        # Two optimizers, one for (conditional )GAN and one for (conditional) DRAW
        vars = tf.trainable_variables()
        vae_vars = [v for v in vars if not v.name.startswith('D/')]
        dis_vars = [v for v in vars if  v.name.startswith('D/')]

        self.vae_opt = tf.train.AdamOptimizer(self.learning_rate).minimize(draw_loss, var_list=vae_vars)
        self.dis_opt = tf.train.AdamOptimizer(self.learning_rate).minimize(dis_loss, var_list=dis_vars)

        self.saver = tf.train.Saver() # saves variables learned during training
        tf.initialize_all_variables().run()


        fetches_draw = []
        fetches_draw.extend([enc_loss, dec_loss, self.vae_opt])

        fetches_gan = []
        fetches_gan.extend([dis_loss, self.dis_opt])

        self.Lxs=[0] * self.train_itrs
        self.Lzs=[0] * self.train_itrs
        self.Lgans=[0] * self.train_itrs


        if load_file is not None:
            # If want to load weight before training:
            self.saver.restore(sess, load_file) 


        # Pint the number of epochs
        print(self.batch_size * self.train_itrs / 60000 )



        for i in range(self.train_itrs):

	    xtrain, ytrain = train_data.next_batch(self.batch_size) # xtrain is (batch_size x img_size)

            if not self.condition:
                self.feed_dict = {self.x : xtrain}
            else:
                self.feed_dict = {self.x : xtrain, self.info: ytrain}


	    enc_loss, dec_loss, _ = sess.run(fetches_draw, self.feed_dict)


            if self.gan:
                dis_loss, _ = sess.run(fetches_gan, self.feed_dict)
            else:
                dis_loss = None


            # Record related loss information during training
	    self.Lxs[i], self.Lzs[i], self.Lgans[i] = dec_loss, enc_loss, dis_loss


	    if i % self.print_itrs  == 0:
                # print len(self.draw_loss), len(self.dis_loss)
		print("iter=%d : reconstruction loss: %f KL divergence %f gan loss: %f" % (i, self.Lxs[i], self.Lzs[i], self.Lgans[i]))

                self.saver.save(sess, 'draw_{}_{}_checkpoint_{}_{}'.format(self.condition, self.gan, self.learning_rate , self.train_itrs))



    def inference(self):

        # This function input a set of images
        # Return the drawed images from last training iterations.
        # Notice: if the object didn't trained yet, it would report error as self.feed_dict is not initialized yet.

        canvases = self.sess.run(self.cs, self.feed_dict)

        canvases = np.array(canvases) # T x batch x img_size
        return canvases

    def generate(self):

        feed_dict = {}

        if self.condition:
            # If condition, input binary-encoded information
            info = np.zeros((self.batch_size, 10))
            info[range(10), range(10)] = 1
            feed_dict[self.info] = info

            for t in range(self.T):
                feed_dict[self.z[t]] = np.random.randn(self.batch_size, self.z_size)

            canvases = self.sess.run(self.cs, feed_dict)
            canvases = np.array(canvases) # T x batch x img_size
            return canvases


        else:
            for t in range(self.T):
                feed_dict[self.z[t]] = np.random.randn(self.batch_size, self.z_size)

            canvases = self.sess.run(self.cs, feed_dict)
            canvases = np.array(canvases) # T x batch x img_size
            return canvases



    def build_DRAW(self):

        self.z = {}
        # Determine if we want to use 'attention' during reading and writing
        read = read_attn if self.read_attn else read_no_attn
        write = write_attn if self.read_attn else write_no_attn
      
        self.x = tf.placeholder(tf.float32,shape=(self.batch_size, self.img_size)) # input (batch_size * img_size)

        if self.condition:
             self.info = tf.placeholder(tf.float32,shape=(self.batch_size, 10)) # input (batch_size * img_size)


        # Initialize variables
        self.cs = [0] * self.T # sequence of canvases
        self.cs_fresh = [0] * self.T #
        self.z_fresh = [0] *  self.T

        mus, logsigmas, sigmas=[0] * self.T,[0] * self.T,[0] * self.T # gaussian params generated by SampleQ. We will need these for computing loss.
        # Initial states
        h_dec_prev = tf.zeros((self.batch_size, self.dec_size))
        enc_state = self.build_encoder.lstm_enc.zero_state(self.batch_size, tf.float32)
        dec_state = self.build_decoder.lstm_dec.zero_state(self.batch_size, tf.float32)


        #!!!
        dec_state_fresh = self.build_decoder.lstm_dec.zero_state(self.batch_size, tf.float32)

        for t in range(self.T):
            self.z_fresh[t] = tf.random_normal((self.batch_size, self.z_size), mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)



        # Construct the unrolled computational graph
        for t in range(self.T):


            # c is the current plot
            ###################################
            c_prev = tf.zeros((self.batch_size, self.img_size)) if t==0 else self.cs[t-1]
            c_prev_fresh = tf.zeros((self.batch_size, self.img_size)) if t==0 else self.cs_fresh[t-1]
            ###################################


            # Error image
            x_hat = self.x - tf.sigmoid(c_prev) 
            # r is the input the the current encoder
            r = read(self.x, x_hat, h_dec_prev, self.A, self.B, self.read_n, self.DO_SHARE, self.eps)


            ###################################
            # Update the hidden encode state and encode using the label information or not.
            if not self.condition:
                h_enc, enc_state = self.build_encoder(enc_state, tf.concat(1,[r, h_dec_prev]), reuse = self.DO_SHARE)
            else:
                h_enc, enc_state = self.build_encoder(enc_state, tf.concat(1,[r, h_dec_prev, self.info]), reuse = self.DO_SHARE)

            
            self.z[t], mus[t], logsigmas[t], sigmas[t]= sampleQ(h_enc, self.DO_SHARE, \
                                                                  self.batch_size, self.z_size)

            ###################################
            # Update the hidden decode state and decode using the label information or not.
            if not self.condition:
                h_dec, dec_state = self.build_decoder(dec_state, self.z[t],  reuse = self.DO_SHARE)
                h_dec_fresh, dec_state_fresh = self.build_decoder(dec_state_fresh, self.z_fresh[t],  reuse = True)
            else:
                h_dec, dec_state = self.build_decoder(dec_state, tf.concat(1, [self.z[t], self.info]),  reuse = self.DO_SHARE)
                h_dec_fresh, dec_state_fresh = self.build_decoder(dec_state_fresh, tf.concat(1, [self.z_fresh[t], self.info]),  reuse = True)


            ###################################
            ###################################
            # Update the canvas.
            self.cs[t] = c_prev + write(h_dec, self.DO_SHARE, self.write_n, self.A, self.B, self.eps, self.batch_size) # store results

            self.cs_fresh[t] = c_prev_fresh + write(h_dec_fresh, True, self.write_n, self.A, self.B, self.eps, self.batch_size)
            ###################################

            h_dec_prev = h_dec
            h_dec_prev_fresh = h_dec_fresh
            self.DO_SHARE = True # from now on, share variables

        ## Compute the loss

        def binary_crossentropy(t , o):
            return -(t * tf.log(o + self.eps) + (1.0 - t) * tf.log(1.0 - o + self.eps))

        # reconstruction term appears to have been collapsed down to a single scalar value (rather than one per item in minibatch)
        x_recons = tf.nn.sigmoid(self.cs[-1])
        x_generates = tf.nn.sigmoid(self.cs_fresh[-1])


        # After computing binary cross entropy, sum across features then take the mean of those sums across minibatches
        Lx = tf.reduce_sum(binary_crossentropy(self.x, x_recons),1) # reconstruction loss
        Lx = tf.reduce_mean(Lx)

        kl_terms=[0] * self.T

        for t in range(self.T):
            mu2 = tf.square(mus[t])
            sigma2 = tf.square(sigmas[t])
            logsigma = logsigmas[t]
            kl_terms[t]=  0.5 * tf.reduce_sum(mu2 + sigma2 - 2 * logsigma, 1) - self.T * .5 # each kl term is (1xminibatch)
            
        KL = tf.add_n(kl_terms) # this is 1xminibatch, corresponding to summing kl_terms from 1:T
        Lz = tf.reduce_mean(KL) # average over minibatches



        # Discriminator for the GAN module
        with tf.variable_scope('D'):

            """
            === TOGGLE ===
            NOTE: if using fully connected, comment out the x_width lines
            """

            if(self.condition):
                self.outdim = 11
            else:
                self.outdim = 1

            # true image
            self.dis_real,self.lth_layer_real = self.build_discriminator(self.x,
                                                                         batch_size = self.batch_size, outdim = self.outdim
                                                    # x_width=self.x_width
            )


            # reconstruction
            self.dis_fake_recon, self.lth_layer_recon  = self.build_discriminator(
                x_recons, reuse=True,
                batch_size = self.batch_size, outdim = self.outdim
                # x_width=self.x_width
                )


            # fresh generation
            self.dis_fake_fresh, _ = self.build_discriminator(x_generates,
                            reuse=True,
                            batch_size = self.batch_size, outdim = self.outdim
            )

        ##  We could use deep feature of GAN to relace the reconstruction loss. However, this does not give good reults
        # self.decoder_loss = tf.reduce_sum(tf.square(self.lth_layer_real - self.lth_layer_recon))

        self.encoder_loss = Lz
        self.decoder_loss = tf.reduce_sum(tf.square(self.x - x_recons))


        if not self.condition:
            self.cross_entropy_real =  -tf.log(self.dis_real+1e-10)
            self.cross_entropy_fresh =   - tf.log(1.-self.dis_fake_fresh+1e-10)
            self.xentropy_fresh_mean = tf.reduce_mean(self.cross_entropy_fresh)
            self.dis_loss = tf.reduce_mean(self.cross_entropy_real + self.cross_entropy_fresh, name='xentropy_mean')


            self.draw_loss =  self.decoder_loss + self.encoder_loss - 5 *  self.xentropy_fresh_mean

        else:

            print(self.info)

            self.label_extra_real  =  tf.concat(1,[tf.Variable(tf.zeros([self.batch_size, 1]),dtype=tf.float32), self.info])
            self.label_extra_fake =  np.column_stack((np.ones((self.batch_size, 1)), np.zeros((self.batch_size, 10))))

            # The truth matrix batch_size by 11 indicator matrix. First column stand for if the image is artificial or not. All other columns stand for the class of the nature image.
            self.cross_entropy_real = tf.nn.softmax_cross_entropy_with_logits(self.dis_real, self.label_extra_real, name='xentropy_real')
            
            self.cross_entropy_fresh = tf.nn.softmax_cross_entropy_with_logits(self.dis_fake_fresh, self.label_extra_fake , name='xentropy_fresh')

            
            self.dis_loss = tf.reduce_mean(self.cross_entropy_real + self.cross_entropy_fresh, name='xentropy_mean')

            self.enc_loss = tf.reduce_mean(self.encoder_loss) 

            self.cross_entropy_fresh_fake = tf.nn.softmax_cross_entropy_with_logits(self.dis_fake_fresh, self.label_extra_real,
                                                                                    name='xentropy_fresh_fake')

            self.g_loss = tf.reduce_mean(self.cross_entropy_fresh_fake) 

            self.draw_loss =  self.enc_loss + self.decoder_loss +  self.g_loss


        return(self.encoder_loss, self.decoder_loss, self.draw_loss, self.dis_loss )
    


