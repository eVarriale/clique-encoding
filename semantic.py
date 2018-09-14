#!/usr/bin/env python3

''' This module encloses functions that deal with the semantic representation
of the bars problem in the network'''

import copy
import pdb
import numpy as np                  # for math

def a_bar(bar_size, which_bars, ravel=True):
    ''' Input from bar_number-th bar in with size bar_size
    first bar_size bars are vertical, then horizontal'''

    sensory_matrix = np.zeros((bar_size, bar_size))
    for bar_number in which_bars:
        if 0 <= bar_number < bar_size:
            sensory_matrix[:, bar_number] = 1
        else:
            sensory_matrix[bar_number % bar_size, :] = 1

    if ravel:
        sensory_matrix = sensory_matrix.ravel()

    return sensory_matrix

def a_clique_response(clique, graph, external_weights):
    '''compute clique averaged afferent signals for every pattern
    clique: a list of neurons belonging to the same clique
    graph: Graph instance, with info about the network
    external_weights: sensory weights'''
    input_size = external_weights.shape[1]
    bar_size = int(np.sqrt(input_size))
    n_pattern = 2 * bar_size
    response = []
    for pattern in range(n_pattern):
        sum_response = 0
        for neuron in clique:
            sum_response += np.dot(external_weights[neuron, :], a_bar(bar_size, [pattern]))
        response.append(sum_response)
    response = np.array(response)
    response /= graph.s_c
    return response

def clique_responses(graph, external_weights):
    ''' compute every clique response to every pattern, and the most responsive
    cliques '''
    response_list = []
    for clique in graph.clique_list:
        response_list.append(a_clique_response(clique, graph, external_weights))
    response_array = np.array(response_list)
    most_responding = np.argmax(response_array, axis=0)
    '''
    resp2 = copy.copy(response_array)
    for clique, most in enumerate(most_responding):
        resp2[clique, most] = 0
    most_responding2 = np.argmax(response_array, axis=0)
    '''
    return response_array, most_responding

def a_receptive_field(clique, graph, external_weights):
    ''' returns the clique mean sensory weight
    clique: list of neurons belonging to a clique'''
    bar_size = int(np.sqrt(external_weights.shape[1]))
    #recep_field = np.zeros((bar_size, bar_size))
    recep_field = external_weights[clique, :].sum(axis=0)
    recep_field /= graph.s_c
    recep_field = np.reshape(recep_field, (bar_size, bar_size))
    return recep_field, recep_field.min(), recep_field.max()

def receptive_fields(graph, external_weights):
    ''' compute receptive fields of every clique '''
    recep_fields = []

    for i, clique in enumerate(graph.clique_list):
        recep_field, rf_min, rf_max = a_receptive_field(clique, graph, external_weights)
        if i == 0:
            glob_min = rf_min
            glob_max = rf_max
        if glob_min > rf_min:
            glob_min = rf_min
        if glob_max < rf_max:
            glob_max = rf_max
        recep_fields.append(recep_field)
    return recep_fields, glob_min, glob_max
