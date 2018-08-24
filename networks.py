#!/usr/bin/env python3

import numpy as np                  # for math
import networkx as nx               # for graphs
import scipy.linalg as LA           # for linear algebra


def ErdosRenyi(PROB_CONN, NEURONS):
    '''create a random excitatory network, and then complete the graph with inhibitory links'''
    exc_adj = np.triu(np.random.rand(NEURONS, NEURONS) < PROB_CONN, k = 1) # upper triangular matrix above diagonal
    #exc_adj += exc_adj.T # symmetrize 
    inh_adj = np.triu(1 - exc_adj, k = 1)
    
    exc_weights = exc_adj * np.triu( np.random.normal(w_mean, 0.1*w_mean, (NEURONS, NEURONS)) )
    inh_weights = inh_adj * np.triu( np.random.normal(w_mean, 0.1*w_mean, (NEURONS, NEURONS)) )
    weights = (exc_adj - inh_adj) * np.random.normal(w_mean, 0.1*w_mean, (NEURONS, NEURONS))

    #weights = np.random.normal(0, 0.5, (NEURONS, NEURONS))
    weights += weights.T
    np.fill_diagonal(weights, 0)
    G = nx.Graph(weights, weight = weights)
    #pdb.set_trace()
    G_exc = nx.Graph(exc_adj+exc_adj.T)
    return weights, G


def geometric(n_clique = 3, clique_size = 3, sparse = False, w_mean = 1):
    neurons = n_clique * clique_size
    single_clique = np.ones((clique_size, clique_size))/(clique_size - 1)
    exc_adj = LA.block_diag(*([single_clique]*n_clique))
    for i in range(n_clique):
        a, b = i*clique_size, i*clique_size-1
        exc_adj[a, b] = 1 / (clique_size)
        exc_adj[b, a] = 1 / (clique_size)
    exc_link_prob = exc_adj.sum()/(neurons*(neurons-1))
    if sparse:
        inh_adj = (1 - exc_adj) * (np.random.rand(neurons, neurons) < exc_link_prob) # sparse inhibitory connections with P_inh = P_exc
    else:
        inh_adj = (1 - exc_adj/exc_adj)
        inh_adj[np.isnan(inh_adj)] = 1 / clique_size
    inh_adj = np.triu(inh_adj, k = 1)
    inh_adj += inh_adj.T
    weights = w_mean * (exc_adj  - inh_adj)
    np.fill_diagonal(weights, 0.)
    G = nx.Graph(weights, weight = weights)
    G.n_c = n_clique
    G.s_c = clique_size
    G.sparse = sparse

    exc_weights = weights * (weights > 0)
    inh_weights = weights * (weights < 0)
    clique_list = [[i*clique_size + j for j in range(clique_size)] for i in range(n_clique)]
    G.clique_list = clique_list
    return exc_weights, inh_weights, G


def ring(neurons, clique_size = 2, sparse = False, w_exc = 0.4, w_inh = 1):
    w_jk, z_jk, G = geometric(neurons//2, clique_size, sparse, 1)
    G.n_c = neurons
    G.s_c = 1
    w_jk *= w_exc
    z_jk *= w_inh
    clique_list = [[i] for i in range(neurons)]
    G.clique_list = clique_list
    return w_jk, z_jk, G


def rotating_clique(n_clique = 3, clique_size = 3, sparse = False, w_mean = 1):
    neurons = n_clique * clique_size
    exc_links = neurons * (clique_size +1) // 2
    tot_links = neurons*(neurons-1)//2
    A = np.zeros((neurons, neurons))

    # create excitatory intra-clique synapses
    clique_list = [[i + j * n_clique for j in range(clique_size)] for i in range(n_clique)]
    for clique in range(n_clique):
        for i in range(clique_size):
            first = clique + i * n_clique
            for j in range(i+1, clique_size):
                second = clique + j * n_clique
                A[first, second] = 1

    # create inter-clique excitatory synapses            
    np.fill_diagonal(A[:, 1:], 1)
    A[0, -1] = 1

    # 
    if sparse:
        inh_adj = (1 - A) * ( np.random.rand(neurons, neurons) < exc_links / (tot_links - exc_links) ) # sparse inhibitory connections with <#inh> = #exc 
    else:
        inh_adj = (1 - A)
    inh_adj = np.triu(inh_adj, k = 1)

    weights = w_mean * ( A / (clique_size - 1) - inh_adj / clique_size ) #/np.sqrt(clique_size + 2)
    #weights = w_mean * (A - inh_adj) / np.sqrt(clique_size + 2)
    np.fill_diagonal(weights[:, 1:], 1/clique_size)
    weights[0, -1] = 1/clique_size
    weights += weights.T
    np.fill_diagonal(weights, 0.)
    #pdb.set_trace()
    G = nx.Graph(weights, weight = weights)
    G.n_c = n_clique
    G.s_c = clique_size
    G.sparse = sparse
    G.clique_list = clique_list
    
    exc_weights = weights * (weights > 0)
    inh_weights = weights * (weights < 0)
    return exc_weights, inh_weights, G


def external_weights(input_size, neurons, P_bars):
    if input_size == 0: 
        input_size = 1
    avg_active_inputs = P_bars * (2 - P_bars) * input_size
    weights = np.random.normal(1, 0.1, (neurons, input_size))  #/ (0.1 * n_clique**2 )
    weights /= 2 * avg_active_inputs
    if input_size == 1: weights *= 2
    return weights
