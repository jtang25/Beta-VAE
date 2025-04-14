def build_decoder(output_shape, z_dim):
    # Determine the size of the latent feature map (here 16x16 with 128 channels)
    initial_shape = (16, 16, 128)
    initial_units = np.prod(initial_shape)
    
    inputs = tf.keras.Input(shape=(z_dim,))
    x = layers.Dense(initial_units, activation='relu')(inputs)
    x = layers.Reshape(initial_shape)(x)
    x = layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', activation='relu')(x)  # 32x32
    x = SEBlock(128)(x)
    x = layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', activation='relu')(x)   # 64x64
    x = SEBlock(64)(x)
    x = layers.Conv2DTranspose(32, kernel_size=4, strides=2, padding='same', activation='relu')(x)   # 128x128
    x = SEBlock(32)(x)
    outputs = layers.Conv2D(1, kernel_size=3, padding='same', activation='sigmoid')(x)
    
    decoder = models.Model(inputs, outputs, name='decoder')
    return decoder
