import numpy as np
import tensorflow as tf
from GraphOps import generate_graphs


np.random.seed(1)
tf.random.set_seed(1)

num_nodes = 28
probabilities = [.5]
num_graph_copies = 1000
graph_file = None
training_features = generate_graphs(num_nodes, probabilities, num_graph_copies, graph_file)

batch_size = 28
epochs = 10
learning_rate = 1e-2
intermediate_dim = 64
original_dim = num_nodes ** 2
l1_reg_const = 0

# (training_features, _), _ = tf.keras.datasets.mnist.load_data()
# # print(training_features.shape)
# # exit(0)
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
            # activation=tf.nn.sigmoid
            activation=tf.nn.relu
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
            # activation=tf.nn.sigmoid
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


autoencoder = Autoencoder(
    intermediate_dim=intermediate_dim,
    original_dim=original_dim
)
opt = tf.optimizers.Adam(learning_rate=learning_rate)


def loss(model, original, l1_reg):
    reconstruction_error = tf.reduce_mean(tf.square(model(original) - original)) \
                           + l1_reg * tf.reduce_sum(tf.abs(model(original) - original))
    return reconstruction_error


def train(loss, model, opt, original, l1_reg):
    with tf.GradientTape() as tape:
        gradients = tape.gradient(loss(model, original, l1_reg), model.trainable_variables)
    gradient_variables = zip(gradients, model.trainable_variables)
    opt.apply_gradients(gradient_variables)


writer = tf.summary.create_file_writer('tmp')
to_record = False

with writer.as_default():
    with tf.summary.record_if(to_record):
        for epoch in range(epochs):
            print("Epoch number %d" % epoch)
            for step, batch_features in enumerate(training_dataset):
                # if epoch == 9:
                #     print(batch_features)
                #     print(autoencoder(batch_features))
                # exit(0)
                train(loss, autoencoder, opt, batch_features, l1_reg_const)  # , l1_reg_const)
                loss_values = loss(autoencoder, batch_features, l1_reg_const)  # , l1_reg_const)
                original = tf.reshape(batch_features, (batch_features.shape[0], num_nodes, num_nodes, 1))
                reconstructed = tf.reshape(autoencoder(tf.constant(batch_features)),
                                           (batch_features.shape[0], num_nodes, num_nodes, 1))
                tf.summary.scalar('loss', loss_values, step=step)
                tf.summary.image('original', original, max_outputs=10, step=step)
                tf.summary.image('reconstructed', reconstructed, max_outputs=10, step=step)
            tf.print(loss_values)
            # print(loss_values)
            # print(loss_values.numpy())
