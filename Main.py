import numpy as np
import tensorflow as tf
from tqdm import tqdm, trange
from GraphOps import *
from AE_Graph import *
from MultiLayerAE import *


def main():
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    # tf.keras.backend.set_floatx('float64')

    num_nodes = 28
    # probabilities = [.5, .6, .7, .8, .9]
    probabilities = [.75]
    num_graphs = 100
    graph_file = None
    adj_mat_file = 'graphs/adj_mat.npy'
    out_mat_file = 'graphs/out_mat.npy'
    save_graphs = False

    # adj_matrices, laplacians = generate_graphs(num_nodes, probabilities, num_graphs)
    adj_matrices, laplacians = generate_ba_graphs(num_nodes, 20, num_graphs)

    batch_size = 28
    epochs = 6
    learning_rate = 1e-3
    intermediate_dim = 64
    original_dim = num_nodes ** 2
    l1_reg_const = 0

    training_features = adj_matrices

    training_features = training_features / np.max(training_features)
    training_features = training_features.reshape(training_features.shape[0],
                                                  training_features.shape[1] * training_features.shape[2])
    training_features = training_features.astype('float32')

    training_dataset = tf.data.Dataset.from_tensor_slices(training_features)
    training_dataset = training_dataset.batch(batch_size)
    training_dataset = training_dataset.shuffle(training_features.shape[0])
    training_dataset = training_dataset.prefetch(batch_size * 4)

    autoencoder = Autoencoder(
        intermediate_dim=intermediate_dim,
        original_dim=original_dim
    )
    opt = tf.optimizers.Adam(learning_rate=learning_rate)

    for epoch in range(epochs):
        print("Epoch number %d" % epoch)
        for step, batch_features in enumerate(training_dataset):
            # if epoch == 9:
            #     print(batch_features)
            #     print(autoencoder(batch_features))
            # exit(0)
            # print(batch_features)
            train(loss, autoencoder, opt, batch_features, l1_reg_const)
            loss_values = loss(autoencoder, batch_features, l1_reg_const)
            original = tf.reshape(batch_features, (batch_features.shape[0], num_nodes, num_nodes))
            reconstructed = tf.reshape(autoencoder(tf.constant(batch_features)),
                                       (batch_features.shape[0], num_nodes, num_nodes))
            # original = tf.reshape(batch_features, (batch_features.shape[0], num_nodes, num_nodes, 1))
            # reconstructed = tf.reshape(autoencoder(tf.constant(batch_features)),
            #                            (batch_features.shape[0], num_nodes, num_nodes, 1))
            # tf.summary.scalar('loss', loss_values, step=step)
            # tf.summary.image('original', original, max_outputs=10, step=step)
            # tf.summary.image('reconstructed', reconstructed, max_outputs=10, step=step)
        tf.print(loss_values)

    cols = adj_matrices.reshape(adj_matrices.shape[0], adj_matrices.shape[1] * adj_matrices.shape[2])
    # print(adj_matrices.shape, cols.shape)
    outputs = tf.reshape(autoencoder(tf.constant(cols)), (cols.shape[0], num_nodes, num_nodes))
    outputs = np.array([out.numpy() for out in outputs])
    # print(outputs.shape)

    if save_graphs:
        with open(adj_mat_file, 'wb') as f:
            np.save(f, adj_matrices)

        with open(out_mat_file, 'wb') as f:
            np.save(f, outputs)

    ratio_file = 'out_ratios.npy'
    values = []

    for i in trange(cols.shape[0], desc='edge expansions: '):
        original = adj_matrices[i]
        new = outputs[i]
        # print(new)
        # print(adj_mat_to_norm_laplacian(new))
        # print(new)
        orig_edge_expansion = estimate_edge_expansion(adj_mat_to_norm_laplacian(original))
        # new_edge_expansion = estimate_edge_expansion(adj_mat_to_norm_laplacian(new))
        new_edge_expansion = estimate_output_edge_expansion(original, new, num_trials=10)
        benchmark_expansions = generate_benchmark_expansions(original, expected_num_edges(original, new))
        result = np.array([np.sum(original), expected_num_edges(original, new),
                           orig_edge_expansion[0], orig_edge_expansion[1],
                           new_edge_expansion[0], new_edge_expansion[1],
                           benchmark_expansions[0], benchmark_expansions[1]])
        values.append(result)
        # print(result)

    print(values)
    with open(ratio_file, 'wb') as f:
        np.save(f, values)


def read_ratios(file):
    ratio_file = file
    # result = np.array([])
    with open(ratio_file, 'rb') as f:
        result = np.load(f)
    print(result)


def generate_benchmarks(ratio_file):
    benchmark_file = 'benchmarks.npy'


if __name__ == '__main__':
    main()
    # read_ratios('out_ratios.npy')
    # generate_benchmarks('out_ratios.npy')
