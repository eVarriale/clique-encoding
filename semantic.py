#!/usr/bin/env python3

import numpy as np                  # for math

def a_bar(n, bar, ravel = True):
    ''' Input from bar-th bar in n x n
    first n bars are vertical, then horizontal'''

    y = np.zeros((n, n))
    if 0<=bar<n:
        y[:, bar] = 1
    else:
        y[bar % n, :] = 1

    if ravel:
        inputs = y.ravel()
        return inputs
    else:
        return y

def a_clique_response(clique, G, external_weights):
    '''compute clique averaged afferent signals for every pattern
    clique: a list of neurons belonging to the same clique
    G: Graph instance, with info about the network
    external_weights: sensory weights'''

    response = []
    for pattern in range(G.n_c):
        R = 0
        for i in range(len(clique)):
            R += np.dot(external_weights[clique[i], :], a_bar(G.n_c//2, pattern))
        response.append(R)
    response = np.array(response)
    return response / G.s_c

def clique_responses(G, external_weights):
    bar_size = G.n_c//2
    n_bars = G.n_c
    resp = []
    for clique in G.clique_list:
        resp.append(a_clique_response(clique, G, external_weights))
    resp = np.array(resp)
    most_responding = np.argmax(resp, axis=0)
    return resp, most_responding

def a_receptive_field(clique, G, external_weights):
    bar_size = G.n_c//2
    #F = np.zeros((bar_size, bar_size))
    s = 0
    clique_neur = clique + np.arange(0, G.number_of_nodes(), G.n_c)
    F = external_weights[clique_neur, :].sum(axis=0)
    F /= G.s_c
    F = np.reshape(F, (bar_size, bar_size))
    return F, F.min(), F.max()

def receptive_fields(G, external_weights):
    n_clique = G.n_c
    recep_fields = []

    for i in range(n_clique):
        F, F_min, F_max = a_receptive_field(i, G, external_weights)
        if i==0:
            Min = Max = F.mean()
        if Min > F_min:
            Min = F_min
        if Max < F_max:
            Max = F_max
        recep_fields.append(F)
    return recep_fields, Min, Max
