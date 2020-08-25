#!/usr/bin/env python3
"""
Creates an autoencoder
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates an autoencoder
    :param input_dims: an integer containing the dimensions of the model input
    :param hidden_layers: a list containing the number of nodes for each
    hidden layer in the encoder, respectively
    :param latent_dims: an integer containing the dimensions of the latent
    space representation
    :return: encoder, decoder, auto
        encoder is the encoder model
        decoder is the decoder model
        auto is the full autoencoder model
    """
    input_encoder = keras.Input(shape=(input_dims, ))
    input_decoder = keras.Input(shape=(latent_dims, ))

    # Encoder model
    encoded = keras.layers.Dense(hidden_layers[0],
                                 activation='relu')(input_encoder)
    for enc in range(1, len(hidden_layers)):
        encoded = keras.layers.Dense(hidden_layers[enc],
                                     activation='relu')(encoded)

    # Latent layer
    z_mean = keras.layers.Dense(latent_dims, activation=None)(encoded)
    z_log_sigma = keras.layers.Dense(latent_dims, activation=None)(encoded)

    def sample_z(args):
        """
        Sampling function
        """
        mu, sigma = args
        batch = keras.backend.shape(mu)[0]
        dim = keras.backend.int_shape(mu)[1]
        eps = keras.backend.random_normal(shape=(batch, dim))
        return mu + keras.backend.exp(sigma / 2) * eps

    z = keras.layers.Lambda(sample_z,
                            output_shape=(latent_dims,))([z_mean, z_log_sigma])

    encoder = keras.Model(inputs=input_encoder,
                          outputs=[z, z_mean, z_log_sigma])

    # Decoded model
    decoded = keras.layers.Dense(hidden_layers[-1],
                                 activation='relu')(input_decoder)
    for dec in range(len(hidden_layers) - 2, -1, -1):
        decoded = keras.layers.Dense(hidden_layers[dec],
                                     activation='relu')(decoded)
    last = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)
    decoder = keras.Model(inputs=input_decoder, outputs=last)

    encoder_output = encoder(input_encoder)[0]
    decoder_output = decoder(encoder_output)
    auto = keras.Model(inputs=input_encoder, outputs=decoder_output)

    def vae_loss(x, x_decoded_mean):
        """
        VAE loss function
        """
        xent_loss = keras.backend.binary_crossentropy(x, x_decoded_mean)
        xent_loss = keras.backend.sum(xent_loss, axis=1)
        kl_loss = - 0.5 * keras.backend.mean(
            1 + z_log_sigma - keras.backend.square(z_mean) - keras.backend.exp(
                z_log_sigma), axis=-1)
        return xent_loss + kl_loss

    auto.compile(optimizer='adam', loss=vae_loss)

    return encoder, decoder, auto
