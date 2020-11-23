import numpy as np
import tensorflow as tf
import itertools
from GraphOps import *

import autograd.numpy as numpy
from autograd import jacobian


def plateau_relu(x):
    return tf.nn.relu(-1 * tf.nn.relu(1-x) + 1)


class MultiLayerEncoder(tf.keras.layers.Layer):
    def __init__(self, inter_dim1, inter_dim2, inter_dim3):
        super(MultiLayerEncoder, self).__init__()
        self.hidden_layer1 = tf.keras.layers.Dense(
            units=inter_dim1,
             activation=tf.nn.relu,
            #activation=plateau_relu,
            kernel_initializer='he_uniform'
        )
        self.hidden_layer2 = tf.keras.layers.Dense(
            units=inter_dim2,
            activation=tf.nn.relu,
            #activation=plateau_relu,
            kernel_initializer='he_uniform'
        )
        self.output_layer = tf.keras.layers.Dense(
            units=inter_dim3,
            #activation=plateau_relu
            activation=tf.nn.relu,
            kernel_initializer='he_uniform'
        )

    def call(self, input_features):
        layer1 = self.hidden_layer1(input_features)
        layer2 = self.hidden_layer2(layer1)
        return self.output_layer(layer2)
        #return self.output_layer(input_features)


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
            activation=tf.nn.relu,
            #activation=plateau_relu,
            kernel_initializer='he_uniform'
        )
        self.hidden_layer3 = tf.keras.layers.Dense(
            units=inter_dim1,
            activation=tf.nn.relu,
            #activation=plateau_relu,
            kernel_initializer='he_uniform'
        )
        self.output_layer = tf.keras.layers.Dense(
            units=original_dim,
            #activation=plateau_relu,
            #activation=tf.nn.relu,
            kernel_initializer='he_uniform'
        )
    def call(self, code):
        # layer1 = self.hidden_layer1(code)
        layer1 = code  # quick fix to remove the double bottleneck layer
        layer2 = self.hidden_layer2(layer1)
        layer3 = self.hidden_layer3(layer2)
        return self.output_layer(layer3)
        # can leave this and comment out previous four lines to perform network surgery
        #return self.output_layer(code) 


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


@tf.custom_gradient
def custom_eigenloss(output, original):
    
    def Laplacian_for_grad(A):
        A = A-numpy.diag(numpy.diag(A)) # get rid of diagonal elements
        A = 0.5*(A+numpy.transpose(A)) # symmetrize
        D_vec = numpy.sum(A, 0)
        D12 = numpy.diag(numpy.power(D_vec, -0.5))
        return D12 @ A @ D12
    
    # model_outs = tf.zeros(tf.shape(output))
    model_outs = np.zeros(tf.shape(output), dtype=np.float32)
    #dLdAs = np.zeros(tf.shape(output), dtype=np.float32)
    lambda1_vals = []
    for i in range(tf.shape(output)[0]):
        first_output = output[i]
        first_orig = original[i]
        # print(first_output.shape, i)
        new_size = np.int(first_output.shape[0] ** .5)
        square_output = np.reshape(first_output, (new_size, new_size))
        square_original = np.reshape(first_orig, (new_size, new_size))

        # G = create_graph_from_output(square_output, square_original)
        # G = adj_mat_to_norm_laplacian(G)
        # eigvals, eigvecs = np.linalg.eig(scipy.sparse.csr_matrix.todense(G))
        #G = square_output
        
        #grad_fun = jacobian(Laplacian_for_grad)
        #dLdA = grad_fun(square_output)
        #print(dLdA)
        
#        G = 0.5*(square_output+np.transpose(square_output))
#        
#        epsilon = 1e-2
#        D_vec = np.sum(G, 0)+epsilon
#        D12_vec = np.power(D_vec, -0.5)
#        D32_vec = np.power(D_vec, -1.5)
#        sumd12 = np.outer(D12_vec, D12_vec)
#        D32 = np.array([[di+dj for di in D32_vec] for dj in D32_vec])
#        
#        D = np.diag(D_vec)
#        D12 = np.diag(D12_vec)
#        dLdA = D32 * (G @ D12)- sumd12 * np.ones(G.shape)
#        if np.any(np.isnan(dLdA)):
#           dLdA = np.zeros(dLdA.shape) # zero it out to avoid affecting gradients
#        #print(dLdA)
#        L = adj_mat_to_norm_laplacian(G)
        L = adj_mat_to_norm_laplacian(0.5*(square_output+np.transpose(square_output)))
        
        try:
            eigvals, eigvecs = np.linalg.eig(L)
        except Exception:
            fail_value = -100
            eigvals = np.repeat(fail_value, square_output.shape[0])
            eigvecs = np.ones(square_output.shape)
        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        lambda_1 = eigvals[1]
        lambda1_vals.append(lambda_1)
        l1_eigvec = eigvecs[:, 1]
        eig_outer = np.outer(l1_eigvec, l1_eigvec)
        #gr = np.einsum('ijkl,kl',dLdA, eig_outer)
        gr = np.zeros(eigvecs.shape) 
        
        #eig_outer = np.reshape(eig_outer, len(l1_eigvec) ** 2)
        gr = np.reshape(gr, len(l1_eigvec) ** 2)
        gr = gr-np.diag(np.diag(gr)) # remove diagonal elements
        if np.any(np.isnan(gr)):
            gr = np.zeros(gr.shape)
        model_outs[i] = gr
        #dLdA = np.reshape(eig_outer, len(l1_eigvec) ** 2)
        #dLdAs[i] = dLdA
    model_outs = tf.convert_to_tensor(model_outs)
    #dLdAs = tf.convert_to_tensor(dLdAs)

    def grad(dABydW):
        deBydW = dABydW * model_outs  # this is element wise multiply
        #deBydW = dABydW * dLdAs * model_outs
        return deBydW, None
        # return None, deBydW

    return tf.constant(np.mean(lambda1_vals), dtype=tf.float32), grad


def loss(model, original, l2_const, l1_const, eigen_const):
    # print(original)
    # print(model(original))
    #reconstruction_error = l2_const * tf.reduce_mean(tf.square(model(original) - original)) \
    #                       + l1_const * tf.reduce_mean(tf.abs(model(original))) \
    #                       - eigen_const * custom_eigenloss(model(original), original)
    full_output = model(original)+original
    reconstruction_error = l2_const * tf.reduce_mean(tf.square(model(original))) \
                           + l1_const * tf.reduce_mean(tf.abs(full_output)) \
                           - eigen_const * custom_eigenloss(full_output, original)
                           #+ eigen_const * custom_eigenloss(model(original), original)

    return reconstruction_error


def train(loss, model, opt, original, l2_const, l1_const, eigen_const):
    with tf.GradientTape() as tape:
        gradients = tape.gradient(loss(model, original, l2_const, l1_const, eigen_const), model.trainable_variables)
    # print(tf.shape(gradients))
    gradient_variables = zip(gradients, model.trainable_variables)
    opt.apply_gradients(gradient_variables)
