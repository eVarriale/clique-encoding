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
    single_clique = np.ones((clique_size, clique_size))
    exc_adj = LA.block_diag(*([single_clique]*n_clique))
    for i in range(n_clique):
        a, b = i*clique_size, i*clique_size-1
        exc_adj[a, b] = 1
        exc_adj[b, a] = 1
    exc_link_prob = exc_adj.sum()/(neurons*(neurons-1))
    if sparse:
        inh_adj = (1 - exc_adj) * (np.random.rand(neurons, neurons) < exc_link_prob) # sparse inhibitory connections with P_inh = P_exc
    else:
        inh_adj = (1 - exc_adj) * exc_link_prob/(1 - exc_link_prob) # all-to-all inhibitory connections with balanced weights
    inh_adj = np.triu(inh_adj, k = 1)
    inh_adj += inh_adj.T
    weights = w_mean * (exc_adj - inh_adj)/np.sqrt(clique_size + 2)
    np.fill_diagonal(weights, 0.)
    G = nx.Graph(weights, weight = weights)
    G.n_c = n_clique
    G.s_c = clique_size
    G.sparse = sparse
    return weights, G

def rotating_clique(n_clique = 3, clique_size = 3, sparse = False, w_mean = 1):
    neurons = n_clique * clique_size
    exc_links = neurons * (clique_size +1) // 2
    tot_links = neurons*(neurons-1)//2
    A = np.zeros((neurons, neurons))
    for clique in range(n_clique):
        for i in range(clique_size):
            first = clique + i * n_clique
            for j in range(i+1, clique_size):
                second = clique + j * n_clique
                #print (first, second)
                A[first, second] = 1
    np.fill_diagonal(A[:, 1:], 1) 
    A[0, -1] = 1
    #G = nx.Graph(A)
    #nx.draw_circular(G, with_labels = True) 
    exc_link_prob = exc_links / tot_links
    #A += A.T
    if sparse:
        #inh_adj = (1 - A) * ( np.random.rand(neurons, neurons) < exc_link_prob ) # sparse inhibitory connections with P_inh = P_exc
        inh_adj = (1 - A) * ( np.random.rand(neurons, neurons) < exc_links/(tot_links - exc_links) ) # sparse inhibitory connections with <#inh> = #exc 
    else:
        inh_adj = (1 - A) * exc_link_prob/(1 - exc_link_prob) # all-to-all inhibitory connections with balanced weights
    inh_adj = np.triu(inh_adj, k = 1)
    weights = w_mean * (A - inh_adj)/np.sqrt(clique_size + 2)
    weights += weights.T
    np.fill_diagonal(weights, 0.)
    #pdb.set_trace()
    G = nx.Graph(weights, weight = weights)
    G.n_c = n_clique
    G.s_c = clique_size
    G.sparse = sparse
    return weights, G