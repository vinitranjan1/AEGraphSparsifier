import numpy as np
import tensorflow as tf


def plateau_relu(x):
    return tf.nn.relu(-1 * tf.nn.relu(1-x) + 1)


class MultiLayerEncoder(tf.keras.layers.Layer):
    def __init__(self, inter_dim1, inter_dim2, inter_dim3):
        super(MultiLayerEncoder, self).__init__()
        self.hidden_layer1 = tf.keras.layers.Dense(
            units=inter_dim1,
            # activation=tf.nn.relu,
            activation=plateau_relu,
            kernel_initializer='he_uniform'
        )
        self.hidden_layer2 = tf.keras.layers.Dense(
            units=inter_dim2,
            # activation=tf.nn.relu,
            activation=plateau_relu,
            kernel_initializer='he_uniform'
        )
        self.output_layer = tf.keras.layers.Dense(
            units=inter_dim3,
            activation=plateau_relu
        )

    def call(self, input_features):
        layer1 = self.hidden_layer1(input_features)
        layer2 = self.hidden_layer2(layer1)
        return self.output_layer(layer2)


class MultiLayerDecoder(tf.keras.layers.Layer):
    def __init__(self, inter_dim1, inter_dim2, inter_dim3, original_dim):
        super(MultiLayerDecoder, self).__init__()
        # self.hidden_layer1 = tf.keras.layers.Dense(
        #     units=inter_dim3,
        #     # activation=tf.nn.relu,
        #     activation=plateau_relu,
        #     kernel_initializer='he_uniform'
        # )
        self.hidden_layer2 = tf.keras.layers.Dense(
            units=inter_dim2,
            # activation=tf.nn.relu,
            activation=plateau_relu,
            kernel_initializer='he_uniform'
        )
        self.hidden_layer3 = tf.keras.layers.Dense(
            units=inter_dim1,
            # activation=tf.nn.relu,
            activation=plateau_relu,
            kernel_initializer='he_uniform'
        )
        self.output_layer = tf.keras.layers.Dense(
            units=original_dim,
            activation=plateau_relu
        )

    def call(self, code):
        # layer1 = self.hidden_layer1(code)
        layer1 = code  # quick fix to remove the double bottleneck layer
        layer2 = self.hidden_layer2(layer1)
        layer3 = self.hidden_layer3(layer2)
        return self.output_layer(layer3)


class MultiLayerAutoencoder(tf.keras.Model):
    def __init__(self, inter_dim1, inter_dim2, inter_dim3, original_dim):
        super(MultiLayerAutoencoder, self).__init__()
        self.encoder = MultiLayerEncoder(
            inter_dim1=inter_dim1,
            inter_dim2=inter_dim2,
            inter_dim3=inter_dim3
        )
        self.decoder = MultiLayerDecoder(
            inter_dim1=inter_dim1,
            inter_dim2=inter_dim2,
            inter_dim3=inter_dim3,
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
