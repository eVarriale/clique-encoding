#!/usr/bin/env python3

import numpy as np                  # for math
import matplotlib.pyplot as plt     # for plots
import networkx as nx               # for graphs
from cycler import cycler           # for managing plot colors
from matplotlib import cm
import matplotlib.colors as clr
import matplotlib.gridspec as gridspec
import semantic
from math import log10, floor
import pdb

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

def network(G):
    exc_edge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 0.]
    inh_edge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] < 0.]
    first_clique = G.clique_list[0]
    clique_size = len(first_clique)
    clique_edge = [(first_clique[i], first_clique[j]) for i in range(clique_size) for j in range(i+1, clique_size) ]
    fig_net, ax_net = plt.subplots()
    pos = nx.circular_layout(G)
    #pos = nx.shell_layout(G)
    #pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos = pos, ax = ax_net) # 
    nx.draw_networkx_edges(G, pos, edgelist=exc_edge) #
    nx.draw_networkx_edges(G, pos, edgelist=inh_edge, ax = ax_net, style = 'dashed', edge_color = 'b', alpha=0.5)
    nx.draw_networkx_edges(G, pos, edgelist=clique_edge, ax = ax_net, edge_color='red') #
    nx.draw_networkx_labels(G, pos) 
    labels = nx.get_edge_attributes(G,'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    if G.sparse:
        inhibition = 'Random inhibitory connections'
    else:
        inhibition = 'Full inhibitory background'
    plt.suptitle('Network of {} cliques with {} nodes'.format(G.n_c, G.s_c))
    plt.title(inhibition)
    ax_net.axis('off')
    return fig_net, ax_net

def rotating_cycler(n_clique):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    #rotating_cycler = cyclesr('linestyle', ['-', '--', ':', '-.']) * cycler(color = colors)
    rotating_cycler = cycler(color = colors)
    rotating_cycler = rotating_cycler[:n_clique]
    return rotating_cycler

def color_map(n_clique):
    start = 0.
    stop = 0.9
    cm_subsection = np.linspace(start, stop, n_clique)
    colors = [ cm.viridis(x) for x in cm_subsection]
    return colors

def activity(G, time_plot, neurons_plot, Y_plot, A, B, gain_rule, save_figures):
    fig, ax = plt.subplots()
    fig_title = 'Activity of {} cliques with {} nodes, sparse = {}'.format(G.n_c, G.s_c, G.sparse)
    if gain_rule == 1:
        subtitle = 'Target variance'# = {:.2f}'.format(TARGET_VAR)
    elif gain_rule == 2:
        subtitle = 'Self-organized variance'
    elif gain_rule == 3:
        subtitle = 'Gain is fixed'

    plt.suptitle(fig_title)
    plt.title(subtitle)
    ax.set(ylabel = 'Activity y', xlabel = 'time (s)')
    cycler = rotating_cycler(G.n_c)
    ax.set_prop_cycle(cycler)
    ax.set_xlim(time_plot[-4999], time_plot[-1])

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    #colors = color_map(G.n_c)
    for i in range(neurons_plot):
        line, = ax.plot(time_plot, Y_plot[i].T, label = i if i<G.n_c else '_nolegend_', color = colors[i%G.n_c])
        #derivative = ( (Y_plot[i, 1:]-Y_plot[i, :-1])/(time_plot[1] - time_plot[0]) ).T
        #deriv, = ax.plot(time_plot[:-1], derivative, label = i if i<G.n_c else '_nolegend_', color = colors[i%G.n_c], linestyle = '-.')

        line_color = line.get_color()
        if (A!=0).any():
            ax.plot(time_plot, A[i].T, label = 'gain {}'.format(i) if i==0 else '_nolegend_', color = colors[i%G.n_c], linestyle = '-.')
            ax.plot(time_plot, B[i].T, label = 'threshold {}'.format(i) if i==0 else '_nolegend_', color = colors[i%G.n_c], linestyle = '--')
        #ax.plot([time_plot[0], time_plot[-1]], [mean, mean], '--')
        #std_line, = ax.plot([time_plot[0], time_plot[-1]], [mean + std, mean + std], '--')
        #ax.plot([time_plot[0], time_plot[-1]], [mean - std, mean - std], '--', color = std_line.get_color())
    ax.legend()
    if save_figures:
        savefig_options = {'papertype' : 'a5', 'dpi' : 180}
        fig.savefig('{}x{}_s{}_r{}'.format(G.n_c, G.s_c, G.sparse, gain_rule), **savefig_options)
        #plt.close(fig)
        #Y = 0
    return fig, ax

def input(G, time_plot, neurons_plot, input_pl, save_figures):
    fig_title_inp = 'Input of {} cliques with {} nodes, sparse = {}'.format(G.n_c, G.s_c, G.sparse)
    fig_inp, ax_inp = plt.subplots()
    ax_inp.set(ylabel = 'Input', xlabel = 'time (s)', title = fig_title_inp)
    cycler = rotating_cycler(G.n_c)
    ax_inp.set_prop_cycle(cycler)
    ax_inp.set_xlim(time_plot[-5000], time_plot[-1])
    for i in range(neurons_plot):
        line, = ax_inp.plot(time_plot, input_pl[i].T, label = i if i<G.n_c else '_nolegend_')
    ax_inp.legend()
    return fig_inp, ax_inp

def response(G, ext_weights, ax = None):
    N_clique = G.n_c
    if ax == None:
        fig, ax = plt.subplots()
    resp, most_responding = semantic.clique_responses(G, ext_weights)
    for i in range(N_clique):
        plt.plot(resp[i], label = 'C {}'.format(i))
    #ax.legend(frameon=False)
    plt.suptitle('Clique C response to patterns P, ' + r'$ R(C, P) = \frac{1}{S_C} \sum_{i \in C, \, j} v_{ij} y_j^P $')
    #plt.title(r'$ R(C, P) = \frac{1}{S_C} \sum_{i \in C \, j} v_{ij} y_j^P $')
    ax.set(ylabel = 'Response R', xlabel = 'Pattern P')
    ax.set_xticks(np.arange(G.n_c))
    ax.set_xticklabels([])
    #pdb.set_trace()
    return ax, most_responding

def patterns(G, ext_weights, axs = None):
    if axs == None:
        fig, axs = plt.subplots(1, G.n_c)
    n_patt = G.n_c//2
    for i in range(len(axs)):
        axs[i].matshow(semantic.a_bar(n_patt, i, False), cmap='Greys')
        axs[i].set_xticks(np.arange(-.5, n_patt, 1))
        axs[i].set_yticks(np.arange(-.5, n_patt, 1))
        axs[i].grid(lw=1, c='k')
        axs[i].tick_params(axis='both',  left=False, top=False, right=False, 
            bottom=False, labelleft=False, labeltop=False, labelright=False, 
            labelbottom=False)

        #axs[i].set(xticks = [], yticks = [])
    return axs

def receptive_fields(G, ext_weights, axs = None, fig = None, most_responding = None):
    rec_fields, Min, Max = semantic.receptive_fields(G, ext_weights)
    if axs == None:
        fig, axs = plt.subplots(1, G.n_c)

    if type(most_responding) == type(None):
        loop_over = range(len(axs))
    else:
        loop_over = most_responding
    for i, j in enumerate(loop_over):
        img = axs[i].matshow(rec_fields[j], vmin=Min, vmax=Max)
        axs[i].tick_params(axis='both',  left=False, top=False, right=False, 
            bottom=False, labelleft=False, labeltop=False, labelright=False, 
            labelbottom=False)
        for spine in axs[i].spines.keys(): # cycles over top, bottom, left, right
            axs[i].spines[spine].set(color=colors[j], lw=2)

    #tick_min = round(Min, -int(floor(log10(abs(Min)))))
    #tick_max = round(Max, -int(floor(log10(abs(Max)))))
    #fig.colorbar(img, ax=axs, ticks=[tick_min, tick_max], aspect=5)
    return axs

def complete_figure(G, ext_weights):
    fig = plt.figure()
    grid_rows = 4
    gs = gridspec.GridSpec(grid_rows, G.n_c, height_ratios = [G.n_c, 1, 1, 1])
    ax_resp = plt.subplot(gs[0,:])
    ax_resp, most_responding = response(G, ext_weights, ax_resp)

    axs_bars = []
    for i in range(G.n_c):
        axs_bars.append(plt.subplot(gs[1,i]))
    axs_bars = patterns(G, ext_weights, axs_bars)

    axs_recep_max = []
    for i in range(G.n_c):
        axs_recep_max.append(plt.subplot(gs[2,i]))
    axs_recep_max = receptive_fields(G, ext_weights, axs_recep_max, fig, most_responding)

    axs_recep = []
    for i in range(G.n_c):
        axs_recep.append(plt.subplot(gs[3,i]))
    axs_recep = receptive_fields(G, ext_weights, axs_recep, fig)

    return fig, ax_resp, axs_bars, axs_recep

def external_weights():
    a = 1

