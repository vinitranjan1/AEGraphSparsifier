import numpy as np
import networkx as nx
import scipy
from scipy.sparse import csgraph
from tqdm import tqdm, trange


def generate_graphs(num_nodes, probabilities, num_graph_copies, graph_file=None):
    # returns two lists, the first one is list of adjacency matrices, second one is list of normalized laplacians
    graphs = []
    graph_dir = 'graphs/'
    if graph_file is None:
        for p in probabilities:
            for _ in trange(num_graph_copies):
                graphs.append(nx.erdos_renyi_graph(num_nodes, p))
        # print([nx.to_numpy_matrix(G) for G in graphs])
        # print(np.array([nx.to_numpy_matrix(G) for G in graphs]).shape)
        return np.array([nx.to_numpy_matrix(G) for G in graphs]), \
            np.array([nx.normalized_laplacian_matrix(G) for G in graphs])


def generate_ba_graphs(num_nodes, num_connections, num_graph_copies, graph_file=None):
    # returns two lists, the first one is list of adjacency matrices, second one is list of normalized laplacians
    graphs = []
    graph_dir = 'graphs/'
    if graph_file is None:
        for _ in trange(num_graph_copies):
            graphs.append(nx.barabasi_albert_graph(num_nodes, num_connections))
        # print([nx.to_numpy_matrix(G) for G in graphs])
        # print(np.array([nx.to_numpy_matrix(G) for G in graphs]).shape)
        return np.array([nx.to_numpy_matrix(G) for G in graphs]), \
            np.array([nx.normalized_laplacian_matrix(G) for G in graphs])


def estimate_edge_expansion(norm_laplacian):
    eigvals = np.linalg.eigvals(scipy.sparse.csr_matrix.todense(norm_laplacian))
    # print('vals')
    # print(eigvals)
    # print(np.sort(eigvals))
    lambda_1 = np.sort(eigvals)[1]
    # print(lambda_1)

    edge_exp_lo = np.real(lambda_1/2)
    edge_exp_hi = np.real(np.sqrt(2*lambda_1))
    #edge_exp_lo = lambda_1/2
    #edge_exp_hi = np.sqrt(2*lambda_1)

    return np.array([edge_exp_lo, edge_exp_hi])


def estimate_output_edge_expansion(original, new, num_trials=100):
    lower_bounds = []
    upper_bounds = []
    for _ in range(num_trials):
        temp = create_graph_from_output(original, new)
        expansion = estimate_edge_expansion(adj_mat_to_norm_laplacian(temp))
        lower_bounds.append(expansion[0])
        upper_bounds.append(expansion[1])
    return np.array([np.average(lower_bounds), np.average(upper_bounds)])


def generate_benchmark_expansions(original, new_num_edges, num_trials=10):
    lower_bounds = []
    upper_bounds = []
    for _ in range(num_trials):
        temp = create_random_subgraph(original, new_num_edges)
        expansion = estimate_edge_expansion(adj_mat_to_norm_laplacian(temp))
        lower_bounds.append(expansion[0])
        upper_bounds.append(expansion[1])
    return np.array([np.average(lower_bounds), np.average(upper_bounds)])


def adj_mat_to_norm_laplacian(adj_mat):
    #G = nx.from_numpy_matrix(adj_mat)
    #return nx.normalized_laplacian_matrix(G)
    G = csgraph.csgraph_from_dense(adj_mat)
    return csgraph.laplacian(G, normed=True)


def create_random_subgraph(original, num_new_edges):
    out = np.zeros(original.shape)
    out = out.astype(int)
    ratio = num_new_edges / (np.sum(original)/2)
    for i in range(original.shape[0]):
        for j in range(i+1, original.shape[0]):
            # print(i, j)
            r = np.random.uniform(0, 1)
            # print(r)
            if original[i][j] and r < ratio:
            #if r < ratio:
                out[i][j] = 1
                out[j][i] = 1
    # print(original)
    # print(new)
    # print(out)
    return out


def create_graph_from_output(original, new, edge_function=max):
    out = np.zeros(original.shape)
    out = out.astype(int)
    for i in range(original.shape[0]):
        for j in range(i+1, original.shape[0]):
            # print(i, j)
            r = np.random.uniform(0, 1)
            # print(r)
            # if original[i][j] and r < edge_function(new[i][j], new[j][i]):
            if original[i][j] and r < np.mean((new[i][j], new[j][i])):
            #if r < np.mean((new[i][j], new[j][i])):
                out[i][j] = 1
                out[j][i] = 1
    # print(original)
    # print(new)
    # print(out)
    return out


def expected_num_edges(original, new, edge_function=max):
    out = 0
    for i in range(original.shape[0]):
        for j in range(i+1, original.shape[0]):
            if original[i][j]:
                #out += edge_function(new[i][j], new[j][i])
                out += np.mean([new[i][j], new[j][i]])
    return out


def main():
    num_nodes = 10
    # probabilities = [.5, .6, .7, .8, .9]
    probabilities = [.5]
    num_graph_copies = 100
    graph_file = None
    graphs, laplacians = generate_graphs(num_nodes, probabilities, num_graph_copies, graph_file)

    edge_exp_lo, edge_exp_hi = estimate_edge_expansion(laplacians[0])
    # print(edge_exp_lo, edge_exp_hi)

    x1 = np.array([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 0]])
    x2 = np.array([[.23, 1.759, .546, 0], [.334, 0, .985, .254], [.713, .349, .215, .32], [0, .98, .213, 1]])
    G = create_graph_from_output(x1, x2)
    adj_mat_to_norm_laplacian(G)
    print(expected_num_edges(x1, x2))


if __name__ == '__main__':
    main()

# NOTE that the following is (only?) valid for d regular graphs
# The case for nonregular graphs relies on the laplacian
#
# def adj_matrix_to_transition(adj_matrix):
#     return normalize(adj_matrix, axis=1, norm='l1')
#
#
# def estimate_edge_expansion(adj_matrix):
#     transition_matrix = adj_matrix_to_transition(adj_matrix)
#     eigvals, _ = np.linalg.eig(transition_matrix)
#     max_deg = max([sum(i) for i in adj_matrix])
#     eigvals = sorted(eigvals, reverse=True)
#     second_eig = eigvals[1]
#     edge_exp_lo = (1-second_eig) * max_deg
#     edge_exp_hi = np.sqrt(8*(1-second_eig)) * max_deg
#
#     edge_exp_lo = (1 - second_eig) / 2
#     edge_exp_hi = np.sqrt(2 * (1 - second_eig))
#
#     return edge_exp_lo, edge_exp_hi
