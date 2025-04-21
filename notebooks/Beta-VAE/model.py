import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from SE_Block.SE_Block import squeeze_excitation_block

latent_dim = 2

class Sampling(layers.Layer):
    """
    Samples z from (mean, logvar) via the reparameterization trick.
    """
    def call(self, inputs):
        mean, logvar = inputs
        epsilon = tf.random.normal(shape=tf.shape(mean))
        return mean + tf.exp(0.5 * logvar) * epsilon


def build_encoder(input_shape=(28, 28, 1)) -> keras.Model:
    """
    Constructs the encoder network that outputs (z_mean, z_logvar, z).
    """
    encoder_inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')(encoder_inputs)
    x = squeeze_excitation_block(x)
    x = layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)
    x = squeeze_excitation_block(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation='relu')(x)
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_logvar = layers.Dense(latent_dim, name='z_logvar')(x)
    z = Sampling(name='z')([z_mean, z_logvar])
    return keras.Model(encoder_inputs, [z_mean, z_logvar, z], name='encoder')
