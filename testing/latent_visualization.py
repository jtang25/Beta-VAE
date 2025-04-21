"""
Standalone script to visualize latent traversal after loading a trained VAE.
"""
import numpy as np
import matplotlib.pyplot as plt
from notebooks.Beta_VAE.autoencoder import BetaVAE

vae = BetaVAE(beta=4.0)
vae.load_weights('path/to/weights.h5')
coords = np.array([[i, j] for i in np.linspace(-3,3,5) for j in np.linspace(-3,3,5)])
imgs = vae.decoder.predict(coords)
fig, axes = plt.subplots(5,5, figsize=(10,10))
for idx,(x,y) in enumerate(coords):
    ax = axes[idx//5, idx%5]
    ax.imshow(imgs[idx].squeeze(), cmap='gray')
    ax.axis('off')
plt.show()