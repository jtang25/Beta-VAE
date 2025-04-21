from tensorflow.keras import layers

def squeeze_excitation_block(inputs, ratio: int = 8):
    channels = inputs.shape[-1]
    se = layers.GlobalAveragePooling2D()(inputs)
    se = layers.Dense(channels // ratio, activation='relu')(se)
    se = layers.Dense(channels, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, channels))(se)
    return layers.multiply([inputs, se])
