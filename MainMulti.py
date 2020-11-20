import numpy as np
import scipy.stats as st
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from GraphOps import *
from MultiLayerAE import *


def main():
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    # tf.keras.backend.set_floatx('float64')

    num_nodes = 4
    #num_nodes = 28
    #num_nodes = 100
    # probabilities = [.5, .6, .7, .8, .9]
    probabilities = [.75]
    num_graphs = 100
    #num_graphs = 200
    graph_file = None
    adj_mat_file = 'graphs/adj_mat.npy'
    out_mat_file = 'graphs/out_mat.npy'
    save_graphs = False

    adj_matrices, laplacians = generate_graphs(num_nodes, probabilities, num_graphs)

    #batch_size = 32
    batch_size = 10
    epochs = 50
    learning_rate = 1e-2
    original_dim = num_nodes ** 2
    l2_reg_const = 1
    l1_reg_const = 0
    eigen_const = 0  # should be positive
    # l1_reg_const = 1e-4

    # intermediate_dim = 256
    # inter_dim1 = 256
    # inter_dim2 = 128
    # inter_dim3 = 64
    inter_dim1 = original_dim
    inter_dim2 = original_dim
    inter_dim3 = original_dim

    training_features = adj_matrices

    training_features = training_features / np.max(training_features)
    training_features = training_features.reshape(training_features.shape[0],
                                                  training_features.shape[1] * training_features.shape[2])
    training_features = training_features.astype('float32')

    training_dataset = tf.data.Dataset.from_tensor_slices(training_features)
    training_dataset = training_dataset.batch(batch_size)
    training_dataset = training_dataset.shuffle(training_features.shape[0])
    training_dataset = training_dataset.prefetch(batch_size * 5)

    autoencoder = MultiLayerAutoencoder(
        inter_dim1=inter_dim1,
        inter_dim2=inter_dim2,
        inter_dim3=inter_dim3,
        original_dim=original_dim
    )
    #opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            learning_rate,
            decay_steps=40*(num_graphs//batch_size),
            decay_rate=0.1,
            staircase=True)
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    #opt = tf.keras.optimizers.Adagrad(learning_rate=lr_schedule)

    for epoch in range(epochs):
        print("Epoch number %d" % epoch)
        #print("Learning rate %f" % round(opt.lr.numpy(), 6)) # doesn't actually work
        for step, batch_features in enumerate(training_dataset):
            # if epoch == 9:
            #     print(batch_features)
            #     print(autoencoder(batch_features))
            # exit(0)
            # print(batch_features)
            train(loss, autoencoder, opt, batch_features, l2_reg_const, l1_reg_const, eigen_const)
            loss_values = loss(autoencoder, batch_features, l2_reg_const, l1_reg_const, eigen_const)
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
    #print(adj_matrices.shape, cols.shape)
    outputs = tf.reshape(autoencoder(tf.constant(cols)), (cols.shape[0], num_nodes, num_nodes))
    outputs = np.array([out.numpy() for out in outputs])
    #print(outputs.shape)

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
        #print(original-new)
        # print(adj_mat_to_norm_laplacian(new))
        # print(new)
        orig_edge_expansion = estimate_edge_expansion(adj_mat_to_norm_laplacian(original))
        # new_edge_expansion = estimate_edge_expansion(adj_mat_to_norm_laplacian(new))
        new_edge_expansion = estimate_output_edge_expansion(original, new, num_trials=10)
        benchmark_expansions = generate_benchmark_expansions(original, expected_num_edges(original, new))
        result = np.array([np.sum(original)/2, expected_num_edges(original, new),
                           orig_edge_expansion[0], orig_edge_expansion[1],
                           new_edge_expansion[0], new_edge_expansion[1],
                           benchmark_expansions[0], benchmark_expansions[1]])
        values.append(result)
        # print(result)

    corr = scipy.stats.pearsonr(adj_matrices.flatten(), outputs.flatten())
    #print(values)
    print(f"Correlation between input and output: {corr}")
    pretty_print(values[-10:])
    print(metrics(values))
    ax = sns.kdeplot(adj_matrices.flatten(), bw_method=0.01)
    sns.kdeplot(outputs.flatten(), bw_method=0.01)
    ax.set(xlabel='inputs/outputs', ylabel='density')
    plt.figure()
    ax2 = sns.kdeplot(adj_matrices.flatten()-outputs.flatten(), bw_method=0.01)
    ax2.set(xlabel='error', ylabel='density')
    
    #print(autoencoder.encoder.hidden_layer1.get_weights())
    
    with open(ratio_file, 'wb') as f:
        np.save(f, values)


def pretty_print(values):
    for result in values:
        result = np.round(result, 3)
        print(f"""old # edges: {result[0]}, expected new # edges: {result[1]},
               old expansion lower bound: {result[2]}, old expansion upper bound: {result[3]},
               new expansion lower bound: {result[4]}, new expansion upper bound: {result[5]},
               benchmark lower bound: {result[6]}, benchmark upper bound: {result[7]}""")


def metrics(values):
    # identify performance metrics based on values
    value_arr = np.array(values)
    sparsity = value_arr[:, 0]/value_arr[:, 1]
    expansion_loss = value_arr[:, 2]/value_arr[:, 4]
    benchmark_loss = value_arr[:, 2]/value_arr[:, 6]
    mean_sparsity = np.mean(sparsity)
    mean_expansion_loss = np.mean(expansion_loss) 
    mean_benchmark_loss = np.mean(benchmark_loss)
    sparsity_CI = st.t.interval(0.95, len(values)-1, loc=np.mean(sparsity), scale=st.sem(sparsity))
    expansion_loss_CI = st.t.interval(0.95, len(values)-1, loc=np.mean(expansion_loss), scale=st.sem(expansion_loss))
    benchmark_loss_CI = st.t.interval(0.95, len(values)-1, loc=np.mean(benchmark_loss), scale=st.sem(benchmark_loss))
    #return sparsity_CI, expansion_loss_CI, benchmark_loss_CI
    return mean_sparsity, mean_expansion_loss, mean_benchmark_loss

def read_ratios(file):
    ratio_file = file
    # result = np.array([])
    with open(ratio_file, 'rb') as f:
        result = np.load(f)
    #print(result)
    pretty_print(result)


def generate_benchmarks(ratio_file):
    benchmark_file = 'benchmarks.npy'


if __name__ == '__main__':
    main()
    # read_ratios('out_ratios.npy')
    # generate_benchmarks('out_ratios.npy')
