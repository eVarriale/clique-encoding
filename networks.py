#!/usr/bin/env python3
import pdb
import numpy as np                  # for math
import networkx as nx               # for graphs
import scipy.linalg as LA           # for linear algebra


def set_up_graph(weights, n_clique, clique_size, clique_list, sparse=False):
    graph = nx.Graph(weights, weight=weights)
    graph.n_c = n_clique
    graph.s_c = clique_size
    graph.sparse = sparse
    graph.clique_list = clique_list
    graph.title = '{} cliques with {} nodes'.format(graph.n_c, graph.s_c)
    return graph

def erdos_renyi(p_conn, neurons, w_mean):
    '''create a random excitatory network, and then complete the graph with inhibitory links'''
    adjacency_exc = np.triu(np.random.rand(neurons, neurons) < p_conn, k=1)
    # upper triangular matrix above diagonal
    #adjacency_exc += adjacency_exc.T # symmetrize 
    adjacency_inh = np.triu(1 - adjacency_exc, k=1)
    
    weights = adjacency_exc - adjacency_inh
    weights *= np.random.normal(w_mean, 0.1*w_mean, weights.shape)
    weights += weights.T
    np.fill_diagonal(weights, 0)
    graph = nx.Graph(weights, weight=weights)
    return weights, graph


def geometric_ring(n_clique, clique_size, sparse=False, w_mean=1):
    neurons = n_clique * clique_size
    single_clique = np.ones((clique_size, clique_size))#/(clique_size - 1)
    adjacency_exc = LA.block_diag(*([single_clique]*n_clique))
    for i in range(n_clique):
        a, b = i*clique_size, i*clique_size - 1
        adjacency_exc[a, b] = 1
        adjacency_exc[b, a] = 1
    if sparse:
        exc_link_prob = adjacency_exc.sum()/(neurons*(neurons-1))
        adjacency_inh = (1 - adjacency_exc) * (np.random.rand(neurons, neurons) < exc_link_prob)
        # sparse inhibitory connections with P_inh = P_exc
    else:
        adjacency_inh = (1 - adjacency_exc)# * exc_link_prob/(1 - exc_link_prob)
    adjacency_inh = np.triu(adjacency_inh, k=1)
    adjacency_inh += adjacency_inh.T
    weights = w_mean * (adjacency_exc - adjacency_inh)#/np.sqrt(clique_size + 2)
    np.fill_diagonal(weights, 0.)

    clique_list = [[i*clique_size + j for j in range(clique_size)] for i in range(n_clique)]
    graph = set_up_graph(weights, n_clique, clique_size, clique_list, sparse)

    exc_weights = weights * (weights > 0)
    inh_weights = weights * (weights < 0)
    return exc_weights, inh_weights, graph


def geometric_net(n_clique, sparse=False, w_mean=1):
    clique_size = n_clique - 1
    neurons = n_clique * clique_size
    single_clique = np.ones((clique_size, clique_size))#/(clique_size - 1)
    adjacency_exc = LA.block_diag(*([single_clique]*n_clique))
    offset = 0
    for clique_a in range(n_clique):
        clique_tour = (offset + np.arange(clique_size - clique_a)) % clique_size
        for clique_distance, relative_a in enumerate(clique_tour, 1):
            a = clique_a * clique_size + relative_a % clique_size

            to_clique = (clique_a + clique_distance) % n_clique
            relative_b = (relative_a + 1) % clique_size
            b = to_clique * clique_size + relative_b
            if clique_distance == 1: 
                offset = relative_b + 1
            assert (relative_b - relative_a) % clique_size == 1
            adjacency_exc[a, b] = 1
            adjacency_exc[b, a] = 1
    if sparse:
        exc_link_prob = adjacency_exc.sum()/(neurons*(neurons-1))
        adjacency_inh = (1 - adjacency_exc) * (np.random.rand(neurons, neurons) < exc_link_prob)
        # sparse inhibitory connections with P_inh = P_exc
    else:
        adjacency_inh = 1 - adjacency_exc
    adjacency_inh = np.triu(adjacency_inh, k=1)
    adjacency_inh += adjacency_inh.T
    weights = w_mean * (adjacency_exc /(clique_size - 1) - adjacency_inh / clique_size)
    #weights = w_mean * (adjacency_exc - adjacency_inh)/(clique_size - 1)
    '''
    for clique_a in range(n_clique):
        clique_tour = (offset + np.arange(clique_size - clique_a)) % clique_size
        for clique_distance, relative_a in enumerate(clique_tour, 1):
            a = clique_a * clique_size + relative_a % clique_size

            to_clique = (clique_a + clique_distance) % n_clique
            relative_b = (relative_a + 1) % clique_size
            b = to_clique * clique_size + relative_b
            if clique_distance == 1: 
                offset = relative_b + 1
            assert (relative_b - relative_a) % clique_size == 1
            weights[a, b] /= 2
            weights[b, a] /= 2
    '''  
    np.fill_diagonal(weights, 0.)
    #weights *= np.random.uniform(0.9, 1.1, weights.shape)

    clique_list = [[i*clique_size + j for j in range(clique_size)] for i in range(n_clique)]
    graph = set_up_graph(weights, n_clique, clique_size, clique_list, sparse)

    exc_weights = weights * (weights > 0)
    inh_weights = weights * (weights < 0)
    return exc_weights, inh_weights, graph


def ring(neurons, clique_size=2, sparse=False, w_exc=0.4, w_inh=1):
    w_jk, z_jk, graph = geometric_ring(neurons//2, clique_size, sparse)
    w_jk *= w_exc
    z_jk *= w_inh
    weights = w_jk + z_jk

    clique_list = [[i] for i in range(neurons)]
    graph = set_up_graph(weights, neurons, clique_size, clique_list, sparse)

    return w_jk, z_jk, graph


def rotating_clique_ring(n_clique, clique_size, sparse=False, w_mean=1):
    neurons = n_clique * clique_size
    exc_links = neurons * (clique_size +1) // 2
    tot_links = neurons*(neurons-1)//2
    adjacency_exc = np.zeros((neurons, neurons))

    # create excitatory intra-clique synapses
    for clique in range(n_clique):
        for i in range(clique_size):
            first = clique + i * n_clique
            for j in range(i+1, clique_size):
                second = clique + j * n_clique
                adjacency_exc[first, second] = 1

    # create inter-clique excitatory synapses
    np.fill_diagonal(adjacency_exc[:, 1:], 1)
    adjacency_exc[0, -1] = 1
    assert (np.triu(adjacency_exc, k=1) == adjacency_exc).all(), \
                'Adjacency matrix not upper triangular'
    # 
    if sparse:
        p_link = exc_links / (tot_links - exc_links)
        adjacency_inh = (1 - adjacency_exc) * (np.random.rand(neurons, neurons) < p_link)
        # sparse inhibitory connections with <#inh> = #exc 
    else:
        adjacency_inh = (1 - adjacency_exc)
    adjacency_inh = np.triu(adjacency_inh, k=1)

    weights = w_mean * (adjacency_exc / (clique_size - 1) - adjacency_inh / clique_size)#/np.sqrt(clique_size + 2)
    #weights = w_mean * (adjacency_exc - adjacency_inh) / np.sqrt(clique_size + 2)
    np.fill_diagonal(weights[:, 1:], 1/clique_size)
    weights[0, -1] = 1/clique_size
    weights += weights.T
    np.fill_diagonal(weights, 0.)
    
    #pdb.set_trace()

    clique_list = [[i + j * n_clique for j in range(clique_size)] for i in range(n_clique)]
    graph = set_up_graph(weights, n_clique, clique_size, clique_list, sparse)

    exc_weights = weights * (weights > 0)
    inh_weights = weights * (weights < 0)
    return exc_weights, inh_weights, graph

def rotating_clique_net(n_clique, w_mean=1):
    ''' network from Gros' paper'''
    clique_size = 4
    neurons = n_clique * clique_size // 2
    adjacency_exc = np.zeros((neurons, neurons))

    # create excitatory intra-clique synapses
    for n in range(neurons):
        if n % 2 == 0:
            close_neurons = n + np.array([-2, -1, +1, +2])
            adjacency_exc[n, close_neurons % neurons] = 1

            opposite_neurons = n + neurons//2 + np.array([-1, +1])
            adjacency_exc[n, opposite_neurons % neurons] = 1

        if n % 2 == 1:
            close_neurons = n + np.array([-1, +1])
            adjacency_exc[n, close_neurons % neurons] = 1

            opposite_neurons = n + neurons//2 + np.array([-1, 0, +1])
            adjacency_exc[n, opposite_neurons % neurons] = 1

    assert (adjacency_exc == adjacency_exc.T).all()
    assert (np.diag(adjacency_exc) == np.zeros(neurons)).all()
    
    adjacency_inh = 1 - adjacency_exc
    np.fill_diagonal(adjacency_inh, 0)

    weights = w_mean * (adjacency_exc / (clique_size - 1) - adjacency_inh / clique_size)
    exc_weights = weights * adjacency_exc
    inh_weights = weights * adjacency_inh

    clique_distance = [0, +1, +2, neurons//2 +1]
    clique_list = [[(i + d) % neurons for d in clique_distance] for i in range(0, neurons, 2)]
    graph = set_up_graph(weights, n_clique, clique_size, clique_list, sparse=False)

    return exc_weights, inh_weights, graph


def external_weights(input_size, neurons, p_bars):
    if input_size == 0: 
        input_size = 1
    avg_active_inputs = p_bars * (2 - p_bars) * input_size
    weights = np.random.normal(1, 0.1, (neurons, input_size))#/ (0.1 * n_clique**2 )
    weights /= 2 * avg_active_inputs
    if input_size == 1: 
        weights = np.array([[10.]])
    return weights