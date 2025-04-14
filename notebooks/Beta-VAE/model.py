class BetaVAE(tf.keras.Model):
    def __init__(self, input_shape, z_dim=32, beta=4.0, **kwargs):
        super(BetaVAE, self).__init__(**kwargs)
        self.z_dim = z_dim
        self.beta = beta
        self.encoder = build_encoder(input_shape, z_dim)
        self.decoder = build_decoder(input_shape, z_dim)
    
    def reparameterize(self, mu, logvar):
        epsilon = tf.random.normal(shape=tf.shape(mu))
        return mu + tf.exp(0.5 * logvar) * epsilon

    def call(self, inputs, training=None):
        mu, logvar = self.encoder(inputs)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar
    
    def compute_beta_vae_loss(model, x):
        reconstructed, mu, logvar = model(x)
        # Reconstruction loss (sum over all pixels)
        reconstruction_loss = tf.reduce_sum(
            losses.binary_crossentropy(x, reconstructed), axis=[1,2,3])
        # KL Divergence
        kl_loss = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar), axis=1)
        total_loss = tf.reduce_mean(reconstruction_loss + model.beta * kl_loss)
        return total_loss, tf.reduce_mean(reconstruction_loss), tf.reduce_mean(kl_loss)
