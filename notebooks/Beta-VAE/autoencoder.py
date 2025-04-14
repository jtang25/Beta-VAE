def build_encoder(input_shape, z_dim):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(32, kernel_size=4, strides=2, padding='same', activation='relu')(inputs)  # 64x64 if input is 128x128
    x = SEBlock(32)(x)
    x = layers.Conv2D(64, kernel_size=4, strides=2, padding='same', activation='relu')(x)      # 32x32
    x = SEBlock(64)(x)
    x = layers.Conv2D(128, kernel_size=4, strides=2, padding='same', activation='relu')(x)     # 16x16
    x = SEBlock(128)(x)
    x = layers.Flatten()(x)
    
    # Latent variable outputs
    mu = layers.Dense(z_dim)(x)
    logvar = layers.Dense(z_dim)(x)
    
    encoder = models.Model(inputs, [mu, logvar], name='encoder')
    return encoder
