import time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from notebooks.Beta_VAE.autoencoder import BetaVAE
from brain_tumor_utils.datautils import load_mnist_data


def visualize_latent_control(decoder, coords: np.ndarray):
    imgs = decoder.predict(coords)
    fig, axes = plt.subplots(1, len(coords), figsize=(len(coords)*2, 2))
    for i, img in enumerate(imgs):
        axes[i].imshow(img.squeeze(), cmap='gray')
        axes[i].axis('off')
    plt.show()


def main():
    (x_train, _), (x_test, _) = load_mnist_data()
    vae = BetaVAE(beta=4.0)
    vae.compile(optimizer=Adam())
    vae.fit(x_train, epochs=30, batch_size=128, validation_data=(x_test, None))

    coords = np.array([[0,0],[1,1],[-1,-1],[2,2],[-2,-2]])
    visualize_latent_control(vae.decoder, coords)


if __name__ == '__main__':
    main()