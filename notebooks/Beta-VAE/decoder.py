import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from SE_Block.SE_Block import squeeze_excitation_block

latent_dim = 2

class DenseTranspose(layers.Layer):
    """
    Custom layer performing a transposed Dense operation.
    """
    def __init__(self, dense_layer, activation=None, **kwargs):
        self.dense = dense_layer
        self.activation = keras.activations.get(activation)
        super().__init__(**kwargs)

    def build(self, batch_input_shape):
        self.bias = self.add_weight(
            name='bias', shape=[self.dense.input_shape[-1]], initializer='zeros')
        super().build(batch_input_shape)

    def call(self, inputs):
        z = tf.matmul(inputs, self.dense.weights[0], transpose_b=True)
        return self.activation(z + self.bias)


def build_decoder() -> keras.Model:
    """
    Constructs the decoder network mapping latent z back to image space.
    """
    dense_layer = layers.Dense(7 * 7 * 64, name='dense_up')
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = DenseTranspose(dense_layer, activation='relu', name='dense_transpose')(latent_inputs)
    x = layers.Reshape((7, 7, 64), name='reshape')(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu', name='deconv1')(x)
    x = squeeze_excitation_block(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu', name='deconv2')(x)
    x = squeeze_excitation_block(x)
    outputs = layers.Conv2DTranspose(1, 3, padding='same', activation='sigmoid', name='reconstruction')(x)
    return keras.Model(latent_inputs, outputs, name='decoder')
