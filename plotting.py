#!/usr/bin/env python3

import pdb
import numpy as np                  # for math
import matplotlib.pyplot as plt     # for plots
from matplotlib import cm
import matplotlib.colors as clr
import matplotlib.gridspec as gridspec
import networkx as nx               # for graphs
from cycler import cycler           # for managing plot colors
import semantic

PROP_CYCLE = plt.rcParams['axes.prop_cycle']
COLORS = PROP_CYCLE.by_key()['color']

def network(graph):
    ''' plots the graph using networkx '''
    exc_edge = [(u, v) for (u, v, d) in graph.edges(data=True) if d['weight'] > 0.]
    inh_edge = [(u, v) for (u, v, d) in graph.edges(data=True) if d['weight'] < 0.]
    first_clique = graph.clique_list[0]
    clique_size = len(first_clique)
    clique_edge = [(first_clique[i], first_clique[j]) for i in range(clique_size) for j in range(i+1, clique_size)]
    fig_net, ax_net = plt.subplots()
    pos = nx.circular_layout(graph)
    #pos = nx.shell_layout(graph)
    #pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph, pos=pos)
    nx.draw_networkx_edges(graph, pos, edgelist=exc_edge, lw=10)
    nx.draw_networkx_edges(graph, pos, edgelist=inh_edge, style='dashed', edge_color='b', alpha=0.5, lw=10)
    nx.draw_networkx_edges(graph, pos, edgelist=clique_edge, edge_color='red', lw=10) #
    nx.draw_networkx_labels(graph, pos)
    labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
    if graph.sparse:
        inhibition = 'Random inhibitory connections'
    else:
        inhibition = 'Full inhibitory background'
    plt.suptitle('Network of {} cliques with {} nodes'.format(graph.n_c, graph.s_c))
    plt.title(inhibition)
    ax_net.axis('off')
    return fig_net, ax_net

def rotating_cycler(n_clique):
    ''' returns a cycler that plt can use to assign colors '''
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    #rotating_cycler = cyclesr('linestyle', ['-', '--', ':', '-.']) * cycler(color = colors)
    my_cycler = cycler(color=colors)
    my_cycler = my_cycler[:n_clique]
    return my_cycler

def color_map(n_clique):
    ''' returns n_clique colors out of viridis color map. '''
    start = 0.
    stop = 0.9
    cm_subsection = np.linspace(start, stop, n_clique)
    colors = [cm.viridis(x) for x in cm_subsection]
    return colors

def activity(graph, time_plot, neurons_plot, Y_plot, A, B, gain_rule, save_figures):
    fig, ax = plt.subplots()
    fig_title = 'Activity of {} cliques with {} nodes'.format(graph.n_c, graph.s_c)
    if gain_rule == 1:
        subtitle = 'Target variance'# = {:.2f}'.format(TARGET_VAR)
    elif gain_rule == 2:
        subtitle = 'Self-organized variance'
    elif gain_rule == 3:
        subtitle = 'Gain is fixed'
    elif gain_rule == 0:
        subtitle = 'Full depletion model'

    plt.suptitle(fig_title)
    plt.title(subtitle)
    ax.set(ylabel='Activity y', xlabel='time (s)')
    #cycler = rotating_cycler(graph.n_c)
    #ax.set_prop_cycle(cycler)
    ax.set_xlim(time_plot[-4999], time_plot[-1])

    #colors = color_map(graph.n_c)
    for i in range(neurons_plot):
        line, = ax.plot(time_plot, Y_plot[i].T, label=i if i < graph.n_c else '_nolegend_', color=COLORS[i%graph.n_c])
        #derivative = ( (Y_plot[i, 1:]-Y_plot[i, :-1])/(time_plot[1] - time_plot[0]) ).T
        #deriv, = ax.plot(time_plot[:-1], derivative, label = i if i<graph.n_c else '_nolegend_', color = COLORS[i%graph.n_c], linestyle = '-.')

        line_color = line.get_color()
        if gain_rule != 0:
            ax.plot(time_plot, A[i].T, label='gain {}'.format(i) if i == 0 else '_nolegend_', color=line_color, linestyle='-.')
            ax.plot(time_plot, B[i].T, label='threshold {}'.format(i) if i == 0 else '_nolegend_', color=line_color, linestyle='--')
        #ax.plot([time_plot[0], time_plot[-1]], [mean, mean], '--')
        #std_line, = ax.plot([time_plot[0], time_plot[-1]], [mean + std, mean + std], '--')
        #ax.plot([time_plot[0], time_plot[-1]], [mean - std, mean - std], '--', color = std_line.get_color())
    ax.legend()
    if save_figures:
        savefig_options = {'papertype' : 'a5', 'dpi' : 180}
        fig.savefig('{}x{}_s{}_r{}'.format(graph.n_c, graph.s_c, graph.sparse, gain_rule), **savefig_options)
        #plt.close(fig)
        #Y = 0
    return fig, ax

def input_signal(graph, time_plot, neurons_plot, input_pl):
    fig_title_inp = 'Input of {} cliques with {} nodes, sparse = {}'.format(graph.n_c, graph.s_c, graph.sparse)
    fig_inp, ax_inp = plt.subplots()
    ax_inp.set(ylabel='Input', xlabel='time (s)', title=fig_title_inp)
    my_cycler = rotating_cycler(graph.n_c)
    ax_inp.set_prop_cycle(my_cycler)
    ax_inp.set_xlim(time_plot[-5000], time_plot[-1])
    for i in range(neurons_plot):
        line, = ax_inp.plot(time_plot, input_pl[i].T, label=i if i < graph.n_c else '_nolegend_')
    ax_inp.legend()
    return fig_inp, ax_inp

def response(graph, ext_weights, ax=None, plot_init=False):
    n_clique = graph.n_c
    if ax is None:
        fig, ax = plt.subplots()
    resp, most_responding = semantic.clique_responses(graph, ext_weights)
    #plot_init = False
    #if initial_weights is not None:
    #    init_resp, init_most = semantic.clique_responses(graph, initial_weights)
    #    plot_init = True
    for i in range(n_clique):
        line, = ax.plot(resp[i], label='C {}'.format(i), color=COLORS[i])
        if plot_init:
            line.set(alpha=0.5, ls='--')

    #ax.legend(frameon=False)
    response_formula = r'$ R(C, P) = \frac{1}{S_C} \sum_{i \in C, \, j} v_{ij} y_j^P $'
    ax.set_title('Clique C response to patterns P, ' + response_formula)
    #plt.title(r'$ R(C, P) = \frac{1}{S_C} \sum_{i \in C \, j} v_{ij} y_j^P $')
    ax.set(ylabel='Response R', xlabel='Pattern P')
    ax.set_xticks(np.arange(graph.n_c))
    ax.set_xticklabels([])
    #pdb.set_trace()
    return ax, most_responding

def patterns(graph, axs=None):
    if axs is None:
        fig, axs = plt.subplots(1, graph.n_c)
    n_patt = graph.n_c//2
    for i in range(len(axs)):
        axs[i].matshow(semantic.a_bar(n_patt, i, False), cmap='Greys')
        axs[i].set_xticks(np.arange(-.5, n_patt, 1))
        axs[i].set_yticks(np.arange(-.5, n_patt, 1))
        axs[i].grid(lw=1, c='k')
        axs[i].tick_params(axis='both', left=False, top=False, right=False,
                           bottom=False, labelleft=False, labeltop=False,
                           labelright=False, labelbottom=False)

        #axs[i].set(xticks = [], yticks = [])
    return axs

def receptive_fields(graph, ext_weights, axs=None, most_responding=None):
    rec_fields, Min, Max = semantic.receptive_fields(graph, ext_weights)
    if axs is None:
        fig, axs = plt.subplots(1, graph.n_c)

    if most_responding is None:
        loop_over = range(len(axs))
    else:
        loop_over = most_responding
    for i, j in enumerate(loop_over):
        img = axs[i].matshow(rec_fields[j], vmin=Min, vmax=Max)
        axs[i].tick_params(axis='both', left=False, top=False, right=False,
                           bottom=False, labelleft=False, labeltop=False,
                           labelright=False, labelbottom=False)
        # cycles over top, bottom, left, right
        for spine in axs[i].spines.keys():
            axs[i].spines[spine].set(color=COLORS[j], lw=2)

    #tick_min = round(Min, -int(floor(log10(abs(Min)))))
    #tick_max = round(Max, -int(floor(log10(abs(Max)))))
    #fig.colorbar(img, ax=axs, ticks=[tick_min, tick_max], aspect=5)
    return axs

def complete_figure(graph, ext_weights, initial_weights=None):
    ''' draws a figure with clique response, patterns, and receptive fields'''
    fig_compl = plt.figure()
    grid_rows = 4
    n_pattern = graph.n_c
    grid_spec = gridspec.GridSpec(grid_rows, n_pattern, height_ratios=[graph.n_c, 1, 1, 1])
    ax_resp = plt.subplot(grid_spec[0, :])
    ax_resp, most_responding = response(graph, ext_weights, ax=ax_resp)
    if initial_weights is not None:
        response(graph, initial_weights, ax=ax_resp, plot_init=True)

    axs_bars = []
    axs_recep_max = []
    axs_recep = []
    for i in range(n_pattern):
        axs_bars.append(plt.subplot(grid_spec[1, i]))
        axs_recep_max.append(plt.subplot(grid_spec[2, i]))
        axs_recep.append(plt.subplot(grid_spec[3, i]))

    axs_bars = patterns(graph, axs_bars)
    axs_recep_max = receptive_fields(graph, ext_weights, axs_recep_max, most_responding)
    axs_recep = receptive_fields(graph, ext_weights, axs_recep)

    return fig_compl, ax_resp, axs_bars, axs_recep
