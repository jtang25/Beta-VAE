"""
Defines the custom loss and BetaVAE model class.
"""
import tensorflow as tf
from tensorflow import keras
from .model import build_encoder
from .decoder import build_decoder


def combined_loss(x, x_recon, mean, logvar, beta: float) -> tf.Tensor:
    """
    Novel adaptive loss combining reconstruction and KL divergence.
    """
    rec = keras.losses.binary_crossentropy(x, x_recon)
    rec = tf.reduce_sum(rec, axis=(1, 2, 3))
    kl = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar), axis=1)
    adaptive = tf.exp(-tf.reduce_mean(rec))
    return tf.reduce_mean(rec + beta * adaptive * kl)


class BetaVAE(keras.Model):
    """
    Beta-Variational Autoencoder with custom train step.
    """
    def __init__(self, beta: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.encoder = build_encoder()
        self.decoder = build_decoder()
        self.beta = beta

    def train_step(self, data):
        x = data[0] if isinstance(data, tuple) else data
        with tf.GradientTape() as tape:
            z_mean, z_logvar, z = self.encoder(x)
            x_recon = self.decoder(z)
            loss = combined_loss(x, x_recon, z_mean, z_logvar, self.beta)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {'loss': loss}