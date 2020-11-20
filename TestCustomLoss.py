"""TensorFlow 2.0 implementation of vanilla Autoencoder."""
import numpy as np
import tensorflow as tf
from GraphOps import *


class Encoder(tf.keras.layers.Layer):
    def __init__(self, intermediate_dim):
        super(Encoder, self).__init__()
        self.hidden_layer = tf.keras.layers.Dense(
            units=intermediate_dim,
            activation=tf.nn.relu,
            kernel_initializer='he_uniform'
        )
        self.output_layer = tf.keras.layers.Dense(
            units=intermediate_dim,
            activation=tf.nn.sigmoid
        )

    def call(self, input_features):
        activation = self.hidden_layer(input_features)
        return self.output_layer(activation)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, intermediate_dim, original_dim):
        super(Decoder, self).__init__()
        self.hidden_layer = tf.keras.layers.Dense(
            units=intermediate_dim,
            activation=tf.nn.relu,
            kernel_initializer='he_uniform'
        )
        self.output_layer = tf.keras.layers.Dense(
            units=original_dim,
            activation=tf.nn.relu
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


#for y= Ax, the derivative is: dy/dx= transpose(A)
@tf.custom_gradient
def custom_eigenloss(output, original):
    # model_outs = tf.zeros(tf.shape(output))
    model_outs = np.zeros(tf.shape(output), dtype=np.float32)
    lambda1_vals = []
    for i in range(tf.shape(output)[0]):
        first_output = output[i]
        first_orig = original[i]
        # print(first_output.shape, i)
        new_size = np.int(first_output.shape[0] ** .5)
        square_output = np.reshape(first_output, (new_size, new_size))
        # square_original = np.reshape(first_orig, (new_size, new_size))

        eigvals, eigvecs = np.linalg.eig(square_output)
        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        lambda_1 = eigvals[1]
        lambda1_vals.append(lambda_1)
        l1_eigvec = eigvecs[:, 1]
        eig_outer = np.outer(l1_eigvec, l1_eigvec)
        eig_outer = np.reshape(eig_outer, len(l1_eigvec) ** 2)
        model_outs[i] = eig_outer
    model_outs = tf.convert_to_tensor(model_outs)

    def grad(dABydW):
        deBydW = dABydW * model_outs  # this is element wise multiply
        return deBydW, None

    return tf.constant(np.mean(lambda1_vals), dtype=tf.float32), grad


def loss(model, original):
    # reconstruction_error = tf.reduce_mean(tf.square(tf.subtract(model(original), original))) \
    #                        + f2(model(original), original)
    reconstruction_error = -1 * custom_eigenloss(model(original), original)
    # reconstruction_error = custom_eigenloss(model(original), original)
    # reconstruction_error = tf.constant(-1, dtype=tf.float32)
    return reconstruction_error


def train(loss, model, opt, original):
    with tf.GradientTape() as tape:
        gradients = tape.gradient(loss(model, original), model.trainable_variables)
    gradient_variables = zip(gradients, model.trainable_variables)
    opt.apply_gradients(gradient_variables)


def main():
    # np.random.seed(1)
    # tf.random.set_seed(1)

    print('loading data')
    num_nodes = 28

    graphs = []
    num_graphs = 10
    for _ in range(num_graphs):
        mat = np.random.rand(num_nodes, num_nodes) * 20
        graphs.append(mat.T @ mat)

    graphs = np.array(graphs)

    for g in graphs:
        eigvals, eigvecs = np.linalg.eig(g)
        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        lambda_1 = eigvals[1]
        print(lambda_1)

    batch_size = 32
    epochs = 20
    learning_rate = 1e-4
    original_dim = num_nodes ** 2
    l1_reg_const = 1e-5
    eigen_const = 1e-5  # should be positive

    training_features = graphs

    intermediate_dim = 784

    training_features = training_features / np.max(training_features)
    training_features = training_features.reshape(training_features.shape[0],
                                                  training_features.shape[1] * training_features.shape[2])
    training_features = training_features.astype('float32')

    training_dataset = tf.data.Dataset.from_tensor_slices(training_features)
    training_dataset = training_dataset.batch(batch_size)
    training_dataset = training_dataset.shuffle(training_features.shape[0])
    training_dataset = training_dataset.prefetch(batch_size * 4)

    writer = tf.summary.create_file_writer('tmp')

    autoencoder = Autoencoder(
        intermediate_dim=intermediate_dim,
        original_dim=original_dim
    )
    opt = tf.optimizers.Adam(learning_rate=learning_rate)

    with writer.as_default():
        with tf.summary.record_if(True):
            for epoch in range(epochs):
                print('epoch: ', epoch)
                for step, batch_features in enumerate(training_dataset):
                    train(loss, autoencoder, opt, batch_features)
                    loss_values = loss(autoencoder, batch_features)
                    original = tf.reshape(batch_features, (batch_features.shape[0], 28, 28, 1))
                    reconstructed = tf.reshape(autoencoder(tf.constant(batch_features)),
                                               (batch_features.shape[0], 28, 28, 1))
                    # tf.summary.scalar('loss', loss_values, step=step)
                    # tf.summary.image('original', original, max_outputs=10, step=step)
                    # tf.summary.image('reconstructed', reconstructed, max_outputs=10, step=step)
                tf.print(loss_values)
                for g in reconstructed:
                    square = np.reshape(g, (num_nodes, num_nodes))
                    eigvals, eigvecs = np.linalg.eig(square)
                    idx = np.argsort(eigvals)
                    eigvals = eigvals[idx]
                    eigvecs = eigvecs[:, idx]
                    lambda_1 = eigvals[1]
                    print(lambda_1)


if __name__ == '__main__':
    main()
