import numpy as np
import tensorflow as tf


def plateau_relu(x):
    return tf.nn.relu(-1 * tf.nn.relu(1-x) + 1)


class Encoder(tf.keras.layers.Layer):
    def __init__(self, intermediate_dim):
        super(Encoder, self).__init__()
        self.hidden_layer = tf.keras.layers.Dense(
            units=intermediate_dim,
            # activation=tf.nn.relu,
            activation=plateau_relu,
            kernel_initializer='he_uniform'
        )
        self.output_layer = tf.keras.layers.Dense(
            units=intermediate_dim,
            # activation=tf.nn.sigmoid
            # activation=tf.nn.relu,
            activation=plateau_relu
        )

    def call(self, input_features):
        activation = self.hidden_layer(input_features)
        return self.output_layer(activation)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, intermediate_dim, original_dim):
        super(Decoder, self).__init__()
        self.hidden_layer = tf.keras.layers.Dense(
            units=intermediate_dim,
            # activation=tf.nn.relu,
            activation=plateau_relu,
            kernel_initializer='he_uniform'
        )
        self.output_layer = tf.keras.layers.Dense(
            units=original_dim,
            # activation=tf.nn.sigmoid
            # activation=tf.nn.relu
            activation=plateau_relu
        )

    def call(self, code):
        activation = self.hidden_layer(code)
        return self.output_layer(activation)


class Autoencoder(tf.keras.Model):
    def __init__(self, intermediate_dim, original_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(intermediate_dim=intermediate_dim)
        self.decoder = Decoder(
            intermediate_dim=intermediate_dim,
            original_dim=original_dim
        )

    def call(self, input_features):
        code = self.encoder(input_features)
        reconstructed = self.decoder(code)
        return reconstructed


def loss(model, original, l1_const):
    reconstruction_error = tf.reduce_mean(tf.square(model(original) - original)) \
                           + l1_const * tf.reduce_sum(tf.abs(model(original) - original))  # TODO: replace w/ general const
    return reconstruction_error


def train(loss, model, opt, original, l1_const):
    with tf.GradientTape() as tape:
        gradients = tape.gradient(loss(model, original, l1_const), model.trainable_variables)
    gradient_variables = zip(gradients, model.trainable_variables)
    opt.apply_gradients(gradient_variables)
