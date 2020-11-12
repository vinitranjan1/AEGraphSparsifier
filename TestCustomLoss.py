"""TensorFlow 2.0 implementation of vanilla Autoencoder."""
import numpy as np
import tensorflow as tf
from GraphOps import *

np.random.seed(1)
tf.random.set_seed(1)
batch_size = 12
epochs = 3
learning_rate = 1e-2
intermediate_dim = 64
original_dim = 784

print('loading data')
num_nodes = 28
#num_nodes = 100
# probabilities = [.5, .6, .7, .8, .9]
probabilities = [.75]
num_graphs = 100
graph_file = None
adj_mat_file = 'graphs/adj_mat.npy'
out_mat_file = 'graphs/out_mat.npy'
save_graphs = False

adj_matrices, laplacians = generate_graphs(num_nodes, probabilities, num_graphs)

batch_size = 32
epochs = 3
learning_rate = 1e-3
original_dim = num_nodes ** 2
l1_reg_const = 1e-5
eigen_const = 1e-5 #should be positive
#l1_reg_const = 1e-4

training_features = adj_matrices

intermediate_dim = 64
inter_dim1 = 256
inter_dim2 = 128
inter_dim3 = 64


training_features = training_features / np.max(training_features)
training_features = training_features.reshape(training_features.shape[0],
                                              training_features.shape[1] * training_features.shape[2])
training_features = training_features.astype('float32')

training_dataset = tf.data.Dataset.from_tensor_slices(training_features)
training_dataset = training_dataset.batch(batch_size)
training_dataset = training_dataset.shuffle(training_features.shape[0])
training_dataset = training_dataset.prefetch(batch_size * 4)


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
            activation=tf.nn.sigmoid
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


autoencoder = Autoencoder(
    intermediate_dim=intermediate_dim,
    original_dim=original_dim
)
opt = tf.optimizers.Adam(learning_rate=learning_rate)


def f1(A, x):
    # B = np.outer(A, A)
    y = tf.matmul(A, x, transpose_a=True, name='y')
    return y, y


#for y= Ax, the derivative is: dy/dx= transpose(A)
@tf.custom_gradient
def f2(A, x):
    y, z = f1(A, x)

    def grad(dzByDy): # dz/dy = 2y reaches here correctly.
        print(tf.shape(A), tf.shape(dzByDy))
        # dzByDx = 2 * tf.matmul(tf.expand_dims(A[0], 0), dzByDy)
        # dzByDx = 2 * tf.matmul(A, dzByDy)  # dzbydy is the type of the OUTPUT BELOW
        dzByDx = dzByDy * A
        return dzByDx, None
    # print(tf.shape(y))
    # return y[0][0], grad
    return tf.constant(0, dtype=tf.float32), grad


def loss(model, original):
    reconstruction_error = tf.reduce_mean(tf.square(tf.subtract(model(original), original))) \
                           + f2(model(original), original)
    return reconstruction_error


def train(loss, model, opt, original):
    with tf.GradientTape() as tape:
        gradients = tape.gradient(loss(model, original), model.trainable_variables)
    gradient_variables = zip(gradients, model.trainable_variables)
    opt.apply_gradients(gradient_variables)


writer = tf.summary.create_file_writer('tmp')

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
