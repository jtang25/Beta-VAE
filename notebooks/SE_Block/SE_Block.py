import tensorflow as tf
from tensorflow.keras import layers

class SEBlock(layers.Layer):
    def __init__(self, channels, reduction=16, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.channels = channels
        self.reduction = reduction
        self.global_avg_pool = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Dense(channels // reduction, activation='relu')
        self.fc2 = layers.Dense(channels, activation='sigmoid')

    def call(self, inputs):
        # Global average pooling
        x = self.global_avg_pool(inputs)
        x = self.fc1(x)
        x = self.fc2(x)
        # Reshape to broadcast on the feature map
        x = tf.reshape(x, (-1, 1, 1, self.channels))
        return inputs * x