#!/usr/bin/env python3

import numpy as np                  # for math
import matplotlib.pyplot as plt     # for plots
import networkx as nx               # for graphs
from cycler import cycler           # for managing plot colors

def network(G):
    exc_edge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 0.]
    inh_edge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] < 0.]
    fig_net, ax_net = plt.subplots()
    pos = nx.circular_layout(G)
    #pos = nx.shell_layout(G)
    #pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos = pos, ax = ax_net) # 
    nx.draw_networkx_edges(G, pos, edgelist=exc_edge, ax = ax_net) #
    nx.draw_networkx_edges(G, pos, edgelist=inh_edge, ax = ax_net, style = 'dashed', edge_color = 'b', alpha=0.5)
    nx.draw_networkx_labels(G, pos)
    ax_net.set_title('Network of {} cliques with {} nodes, sparse = {}'.format(G.n_c, G.s_c, G.sparse))
    ax_net.axis('off')

def rotating_cycler(n_clique):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    rotating_cycler = cycler('linestyle', ['-', '--', ':', '-.']) * cycler(color = colors)
    rotating_cycler = rotating_cycler[:n_clique]
    return rotating_cycler

def activity(G, time_plot, neurons_plot, Y_plot, A, B, gain_rule, save_figures):
    fig, ax = plt.subplots()
    fig_title = 'Activity of {} cliques with {} nodes, sparse = {}'.format(G.n_c, G.s_c, G.sparse)
    if gain_rule == 3:
        subtitle = 'Target variance'# = {:.2f}'.format(TARGET_VAR)
    elif gain_rule == 2:
        subtitle = 'Self-organized variance'
    else:
        subtitle = 'Gain is fixed'
    print(subtitle)

    plt.suptitle(fig_title)
    plt.title(subtitle)
    ax.set(ylabel = 'Activity y', xlabel = 'time (s)') #'excitatory fraction: {}, w_mean: {}'.format(EXC_FRAC, w_mean))
    cycler = rotating_cycler(G.n_c)
    ax.set_prop_cycle(cycler)
    ax.set_xlim(time_plot[-5000], time_plot[-1])

    for i in range(neurons_plot):
        line, = ax.plot(time_plot, Y_plot[i].T, label = i if i<G.n_c else '_nolegend_')
        ax.plot(time_plot, A[i].T, label = 'gain {}'.format(i) if i<G.n_c else '_nolegend_', color = line.get_color())
        ax.plot(time_plot, B[i].T, label = 'threshold {}'.format(i) if i<G.n_c else '_nolegend_', color = line.get_color(), linestyle = '--')
        #time.sleep(2)
        #ax.plot([time_plot[0], time_plot[-1]], [mean, mean], '--')
        #std_line, = ax.plot([time_plot[0], time_plot[-1]], [mean + std, mean + std], '--')
        #ax.plot([time_plot[0], time_plot[-1]], [mean - std, mean - std], '--', color = std_line.get_color())
    ax.legend()
    if save_figures:
        savefig_options = {'papertype' : 'a5', 'dpi' : 180}
        fig.savefig('{}x{}_s{}_r{}'.format(G.n_c, G.s_c, G.sparse, gain_rule), **savefig_options)
        #plt.close(fig)
        #Y = 0

def input(G, time_plot, neurons_plot, input_pl, save_figures):
    fig_title_inp = 'Input of {} cliques with {} nodes, sparse = {}'.format(G.n_c, G.s_c, G.sparse)
    fig_inp, ax_inp = plt.subplots()
    ax_inp.set(ylabel = 'Input', xlabel = 'time (s)', title = fig_title_inp) #'excitatory fraction: {}, w_mean: {}'.format(EXC_FRAC, w_mean))
    cycler = rotating_cycler(G.n_c)
    ax_inp.set_prop_cycle(cycler)
    ax_inp.set_xlim(time_plot[-5000], time_plot[-1])
    for i in range(neurons_plot):
        line, = ax_inp.plot(time_plot, input_pl[i].T, label = i if i<G.n_c else '_nolegend_')
    ax_inp.legend()