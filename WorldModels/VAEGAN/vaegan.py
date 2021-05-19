import tensorflow.keras as keras
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
import sys
import pdb



K_SIZE = 5
inner_loss_coef = 1
normal_coef = 0.1
kl_coef = 0.01

def sampling(args):
    mean, logsigma = args
    epsilon = keras.backend.random_normal(shape=keras.backend.shape(mean))
    return mean + tf.exp(logsigma / 2) * epsilon




class VAEGAN(tf.keras.Model):
    def __init__(self,lr=0.0001,args=None):
        super(VAEGAN, self).__init__()
        self.batch_size = args.vae_gan_batch_size
        self.DEPTH = args.vae_gan_DEPTH
        self.LATENT_DEPTH = args.vae_gan_LATENT_DEPTH



        self.E = self.encoder()
        self.G = self.generator()
        self.D = self.discriminator()

        self.E_opt = keras.optimizers.Adam(learning_rate=lr)
        self.G_opt = keras.optimizers.Adam(learning_rate=lr)
        self.D_opt = keras.optimizers.Adam(learning_rate=lr)


    def encoder(self):
        input_E = keras.layers.Input(shape=(64, 64, 3))
        
        X = keras.layers.Conv2D(filters=self.DEPTH*2, kernel_size=K_SIZE, strides=2, padding='same')(input_E)
        X = keras.layers.BatchNormalization()(X)
        X = keras.layers.LeakyReLU(alpha=0.2)(X)

        X = keras.layers.Conv2D(filters=self.DEPTH*4, kernel_size=K_SIZE, strides=2, padding='same')(X)
        X = keras.layers.BatchNormalization()(X)
        X = keras.layers.LeakyReLU(alpha=0.2)(X)

        X = keras.layers.Conv2D(filters=self.DEPTH*8, kernel_size=K_SIZE, strides=2, padding='same')(X)
        X = keras.layers.BatchNormalization()(X)
        X = keras.layers.LeakyReLU(alpha=0.2)(X)
        
        X = keras.layers.Flatten()(X)
        X = keras.layers.Dense(self.LATENT_DEPTH)(X)    
        X = keras.layers.BatchNormalization()(X)
        X = keras.layers.LeakyReLU(alpha=0.2)(X)
        
        mean = keras.layers.Dense(self.LATENT_DEPTH,activation="tanh")(X)
        logsigma = keras.layers.Dense(self.LATENT_DEPTH,activation="tanh")(X)
        latent = keras.layers.Lambda(sampling, output_shape=(self.LATENT_DEPTH,))([mean, logsigma])
        
        kl_loss = 1 + logsigma - keras.backend.square(mean) - keras.backend.exp(logsigma)
        kl_loss = keras.backend.mean(kl_loss, axis=-1)
        kl_loss *= -0.5
        return keras.models.Model(input_E, [latent,kl_loss])

    def generator(self):
        input_G = keras.layers.Input(shape=(self.LATENT_DEPTH,))

        X = keras.layers.Dense(8*8*self.DEPTH*8)(input_G)
        X = keras.layers.BatchNormalization()(X)
        X = keras.layers.LeakyReLU(alpha=0.2)(X)
        X = keras.layers.Reshape((8, 8, self.DEPTH * 8))(X)
        
        X = keras.layers.Conv2DTranspose(filters=self.DEPTH*8, kernel_size=K_SIZE, strides=2, padding='same')(X)
        X = keras.layers.BatchNormalization()(X)
        X = keras.layers.LeakyReLU(alpha=0.2)(X)

        X = keras.layers.Conv2DTranspose(filters=self.DEPTH*4, kernel_size=K_SIZE, strides=2, padding='same')(X)
        X = keras.layers.BatchNormalization()(X)
        X = keras.layers.LeakyReLU(alpha=0.2)(X)
        
        X = keras.layers.Conv2DTranspose(filters=self.DEPTH, kernel_size=K_SIZE, strides=2, padding='same')(X)
        X = keras.layers.BatchNormalization()(X)
        X = keras.layers.LeakyReLU(alpha=0.2)(X)
        
        X = keras.layers.Conv2D(filters=3, kernel_size=K_SIZE, padding='same')(X)
        X = keras.layers.Activation('sigmoid')(X)
        return keras.models.Model(input_G, X)

    def discriminator(self):
        input_D = keras.layers.Input(shape=(64, 64, 3))
        
        X = keras.layers.Conv2D(filters=self.DEPTH, kernel_size=K_SIZE, strides=2, padding='same')(input_D)
        X = keras.layers.LeakyReLU(alpha=0.2)(X)
        
        X = keras.layers.Conv2D(filters=self.DEPTH*4, kernel_size=K_SIZE, strides=2, padding='same')(input_D)
        X = keras.layers.LeakyReLU(alpha=0.2)(X)
        X = keras.layers.BatchNormalization()(X)

        X = keras.layers.Conv2D(filters=self.DEPTH*8, kernel_size=K_SIZE, strides=2, padding='same')(X)
        X = keras.layers.BatchNormalization()(X)
        X = keras.layers.LeakyReLU(alpha=0.2)(X)

        X = keras.layers.Conv2D(filters=self.DEPTH*8, kernel_size=K_SIZE, padding='same')(X)
        inner_output = keras.layers.Flatten()(X)
        X = keras.layers.BatchNormalization()(X)
        X = keras.layers.LeakyReLU(alpha=0.2)(X)
        
        X = keras.layers.Flatten()(X)
        X = keras.layers.Dense(self.DEPTH*8)(X)
        X = keras.layers.BatchNormalization()(X)
        X = keras.layers.LeakyReLU(alpha=0.2)(X)
        
        output = keras.layers.Dense(1)(X)    
        return keras.models.Model(input_D, [output, inner_output])

    @tf.function
    def train_step_vaegan(self,x):
        lattent_r =  tf.random.normal((self.batch_size, self.LATENT_DEPTH))
        with tf.GradientTape(persistent=True) as tape:
            lattent,kl_loss = self.E(x)
            fake = self.G(lattent)
            dis_fake,dis_inner_fake = self.D(fake)
            dis_fake_r,_ = self.D(self.G(lattent_r))
            dis_true,dis_inner_true = self.D(x)


            vae_inner = dis_inner_fake-dis_inner_true
            vae_inner = vae_inner*vae_inner
            
            mean,var = tf.nn.moments(self.E(x)[0], axes=0)
            var_to_one = var - 1
            
            normal_loss = tf.reduce_mean(mean*mean) + tf.reduce_mean(var_to_one*var_to_one)
            
            kl_loss = tf.reduce_mean(kl_loss)
            vae_diff_loss = tf.reduce_mean(vae_inner)
            f_dis_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(tf.zeros_like(dis_fake), dis_fake))
            r_dis_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(tf.zeros_like(dis_fake_r), dis_fake_r))
            t_dis_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(tf.ones_like(dis_true), dis_true))
            gan_loss = (0.5*t_dis_loss + 0.25*f_dis_loss + 0.25*r_dis_loss)
            vae_loss = tf.reduce_mean(tf.abs(x-fake)) 
            E_loss = vae_diff_loss + kl_coef*kl_loss + normal_coef*normal_loss
            G_loss = inner_loss_coef*vae_diff_loss - gan_loss
            D_loss = gan_loss

        E_grad = tape.gradient(E_loss,self.E.trainable_variables)
        G_grad = tape.gradient(G_loss,self.G.trainable_variables)
        D_grad = tape.gradient(D_loss,self.D.trainable_variables)

        del tape
        self.E_opt.apply_gradients(zip(E_grad, self.E.trainable_variables))
        self.G_opt.apply_gradients(zip(G_grad, self.G.trainable_variables))
        self.D_opt.apply_gradients(zip(D_grad, self.D.trainable_variables))

        return [E_loss,G_loss,D_loss],[gan_loss, vae_loss, f_dis_loss, r_dis_loss, t_dis_loss, vae_diff_loss, kl_loss, normal_loss]

if __name__=='__main__':
    
    class Args():
        def __init__(self):
            self.vae_gan_DEPTH = 32
            self.vae_gan_LATENT_DEPTH = 256
            self.vae_gan_batch_size = 4

    args = Args()
    print(args)
    vaegan = VAEGAN(lr=0.0001,args=args)
    E = vaegan.encoder()
    G = vaegan.generator()
    D = vaegan.discriminator()
    x = tf.random.normal([4,64,64,3])
    print(E(x)[0].shape)
    # print(G(tf.random.normal([4,512,])).shape)
    # print(D(x))

    # loss_list = vaegan.train_step_vaegan(x)
    # print(loss_list)