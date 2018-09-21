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
import dynamics

PROP_CYCLE = plt.rcParams['axes.prop_cycle']
COLORS = PROP_CYCLE.by_key()['color']

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
    stop = 1
    cm_subsection = np.linspace(start, stop, n_clique)
    colors = [cm.viridis(x) for x in cm_subsection]
    return colors


def geometric_shell(graph):
    ''' compute positions: cliques at regular angle intervals, and cliques as
    regular shapes'''
    radius = 2 * len(graph.clique_list)
    neuron_radius = 0.4 * len(graph.clique_list)
    npos = {}
    # Discard the extra angle since it matches 0 radians.
    theta_clique = -np.linspace(0, 1, len(graph.clique_list) + 1)[:-1] * 2 * np.pi 
    theta_clique += np.pi/2
    clique_center = radius *  np.column_stack([np.cos(theta_clique), np.sin(theta_clique)])
    for i, clique in enumerate(graph.clique_list):
        theta_neuron = -np.linspace(0, 1, len(clique)+1)[:-1] * 2 * np.pi
        clique_angle = theta_clique[i] - 2*np.pi/len(clique)/2
        turns = (i * (len(clique) - 2)) % len(clique)
        clique_angle -=  turns * 2 * np.pi/len(clique)
        pos = np.column_stack([np.cos(theta_neuron+clique_angle), np.sin(theta_neuron+clique_angle)])
        pos *= neuron_radius
        pos += clique_center[i]
        npos.update(zip(clique, pos))
    return npos

def neurons_to_clique(graph):
    neuron_to_clique = [0]*graph.number_of_nodes()
    for clique, clique_members in enumerate(graph.clique_list):
        for neuron in clique_members:
            neuron_to_clique[neuron] = clique
    return neuron_to_clique

def network(graph):
    ''' plots the graph using networkx '''
    fig_net, ax_net = plt.subplots()
    ax_net.axis('off')

    #pos = nx.circular_layout(graph)
    #pos = nx.shell_layout(graph)
    #pos = nx.spring_layout(graph)
    #pos = nx.kamada_kawai_layout(graph)
    pos = geometric_shell(graph)
    index_list = neurons_to_clique(graph)
    edgecolor = [COLORS[i] for i in index_list]
    color = "#{0:02x}{1:02x}{2:02x}".format(0,97,143)
    nx.draw_networkx_nodes(graph, pos=pos, node_color=color, 
                            edgecolors=edgecolor, linewidths=2)
    #nx.draw_networkx_nodes(graph, pos=pos)
    exc_edge = [(u, v) for (u, v, d) in graph.edges(data=True) if d['weight'] > 0.]
    nx.draw_networkx_edges(graph, pos, edgelist=exc_edge, lw=10)

    #inh_edge = [(u, v) for (u, v, d) in graph.edges(data=True) if d['weight'] < 0.]
    #nx.draw_networkx_edges(graph, pos, edgelist=inh_edge, style='dashed',
    #                       edge_color='b', alpha=0.5, lw=10)

    #clique = graph.clique_list[0]
    #clique_edge = [(a, b) for i, a in enumerate(clique) for b in clique[i+1:]]
    #nx.draw_networkx_edges(graph, pos, edgelist=clique_edge, edge_color='red', lw=10) #

    #nx.draw_networkx_labels(graph, pos)
    #labels = nx.get_edge_attributes(graph, 'weight')
    #nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)

    plt.tight_layout()

    #plt.suptitle(graph.title)
    if graph.sparse:
        inhibition = 'Random inhibitory connections'
    else:
        inhibition = 'Full inhibitory background'
    #plt.title(inhibition)
    #plt.title(graph.title)
    fig_net.set_size_inches([6, 6])

    return fig_net, ax_net



def activity(graph, time_plot, neurons_plot, y_plot, A, B, gain_rule, 
                save_figures=False, bars_time=None):
    fig, ax = plt.subplots()

    #plt.suptitle('Activity of ' + graph.title)

    if gain_rule == 1:
        subtitle = 'Target variance'# = {:.2f}'.format(TARGET_VAR)
    elif gain_rule == 2:
        subtitle = 'Self-organized variance'
    elif gain_rule == 3:
        subtitle = 'Gain is fixed'
    elif gain_rule == 0:
        subtitle = 'Full depletion model'

    #plt.title(subtitle)
    ax.set(ylabel='Activity y', xlabel='time (s)')
    #cycler = rotating_cycler(graph.n_c)
    #ax.set_prop_cycle(cycler)
    ax.set_xlim(time_plot[-4999], time_plot[-1])

    #colors = color_map(graph.n_c)
    lines = []
    labels = []
    for i in range(neurons_plot):
        for clique, clique_members in enumerate(graph.clique_list):
            if i in clique_members: break

        label = clique if i == clique_members[0] else '_nolegend_'
        # TODO overlapping cliques
        color = COLORS[clique]
        #color = colors[clique]
        line, = ax.plot(time_plot, y_plot[i].T, label=label, color=color)
        lines.append(line)
        #derivative = ( (y_plot[i, 1:]-y_plot[i, :-1])/(time_plot[1] - time_plot[0]) ).T
        #deriv, = ax.plot(time_plot[:-1], derivative, label = i if i<graph.n_c else '_nolegend_', color = COLORS[i%graph.n_c], linestyle = '-.')

        if gain_rule != 0:
            label2 = '$\phi_e u_e$' if i == 0 else '_nolegend_'
            line2, = ax.plot(time_plot, A[i].T, label=label2, color=color, linestyle='-.', alpha=0.5)

            label3 = '$\phi_i u_i$' if i == 0 else '_nolegend_'
            line3, = ax.plot(time_plot, B[i].T, label=label3, color=color, linestyle='--', alpha=0.5)
            
            lines.append([line2, line3])
        #ax.plot([time_plot[0], time_plot[-1]], [mean, mean], '--')
        #std_line, = ax.plot([time_plot[0], time_plot[-1]], [mean + std, mean + std], '--')
        #ax.plot([time_plot[0], time_plot[-1]], [mean - std, mean - std], '--', color = std_line.get_color())
    legend1 = plt.legend(title='Cliques', loc=1,frameon=False)
    ax.add_artist(legend1)

    if bars_time is not None and (bars_time != -1).any():
        bars_time = bars_time[-time_plot.size:]
        colors = color_map(5)
        alpha_values = [0.05, 0.3, 0.5, 0.7, 0.7]
        fills = []
        ylim = ax.get_ylim()
        for i, c in enumerate(colors[:-1]):
            fill = ax.fill_between(time_plot, *ylim, where=bars_time == i,
                                   alpha=alpha_values[i], color=c, label=i) #
            fills.append(fill)

        if any(bars_time > i):
            fill = ax.fill_between(time_plot, *ylim, where=bars_time > i,
                                   alpha=alpha_values[i+1], color=colors[i+1],
                                   label='{}+'.format(i))
            fills.append(fill)
        ax.set_ylim(ylim)

        # Create a legend for the first line.
        first_legend = plt.legend(handles=[f for f in fills], title='Bars')#, frameon=False)
    ax.set_yticks([0, 1])
    plt.tight_layout()
    if save_figures:
        savefig_options = {'papertype' : 'a5', 'dpi' : 300}
        fig.savefig('{}x{}_s{}_r{}'.format(graph.n_c, graph.s_c, graph.sparse, gain_rule), **savefig_options)
        #plt.close(fig)
        #Y = 0
    return fig, ax

def full_depletion(time_plot, full_vesicles_inh_plot, vesic_release_inh_plot,
                    full_vesicles_exc_plot, vesic_release_exc_plot):

    fig_fulldep, ax_fulldep = plt.subplots(nrows=2, sharex=True)

    effective_weights_plot_inh = vesic_release_inh_plot * full_vesicles_inh_plot
    ax_fulldep[0].plot(time_plot, full_vesicles_inh_plot.T, label=r'$\varphi_i$')
    ax_fulldep[0].plot(time_plot, vesic_release_inh_plot.T, label='$u_i$')
    ax_fulldep[0].plot(time_plot, effective_weights_plot_inh.T, label=r'$\varphi_i \cdot u_i$')
    u_phi_max = effective_weights_plot_inh[:, 1800:].max()
    ax_fulldep[0].set_yticks([0, 1, u_phi_max, dynamics.U_max])
    ax_fulldep[0].set_yticklabels(['0', '1', '{:2.1f}'.format(u_phi_max), '$U_{max}$'])
    ax_fulldep[0].legend(frameon=False, prop={'size': 15})

    effective_weights_plot_exc = vesic_release_exc_plot * full_vesicles_exc_plot
    ax_fulldep[1].plot(time_plot, full_vesicles_exc_plot.T, label=r'$\varphi_e$')
    ax_fulldep[1].plot(time_plot, vesic_release_exc_plot.T, label='$u_e$')
    ax_fulldep[1].plot(time_plot, effective_weights_plot_exc.T, label=r'$\varphi_e \cdot u_e$')
    u_phi_max = effective_weights_plot_exc[:, 1800:].max()
    ax_fulldep[1].set_yticks([0, 1, u_phi_max, dynamics.U_max])
    ax_fulldep[1].set_yticklabels(['0', '1', '{:2.1f}'.format(u_phi_max), '$U_{max}$'])
    ax_fulldep[0].set(ylim=[-0.02, dynamics.U_max + .02], xlim=[1.8, 3.8])
    ax_fulldep[1].set(ylim=[-0.02, dynamics.U_max + .02], xlim=[1.8, 3.8])
    ax_fulldep[1].set(xlabel='time (s)')
    ax_fulldep[1].legend(frameon=False, prop={'size': 15})

    #ax_fulldep[0].set_xticks([1.8, 2.3, 2.8, 3.3])
    #ax_fulldep[0].set_xticklabels(['0', '0.5', '1', '1.5'])
    plt.tight_layout()
    return fig_fulldep, ax_fulldep

def activity_stripes(graph, y_plot, time_plot):
    neurons = graph.number_of_nodes()
    fig, axs = plt.subplots(neurons, 1, sharex=True)
    plt.suptitle('Activity of ' + graph.title)
    fig.subplots_adjust(hspace=0)
    #color = { 0 : 'xkcd:pale blue', 1 : 'xkcd:white'}
    #color_switch = True
    clique_mean = y_plot[graph.clique_list, :].mean(axis=1)

    winning_clique = np.argmax(clique_mean, axis=0)
    for i in range(neurons):
        #if(i % clique_size==0):
        #    color_switch = not color_switch
        axs[i].set_ylim(-.05, 1.05)
        axs[i].plot(time_plot, y_plot[i], label=i)
        #axs[i].fill_between(time_plot, 0, 1, color=COLORS[winning_clique]
        #axs[i].plot(time_plot, B[i, -final_times:], 'r--', linewidth=1)
        #axs[i].set_facecolor(color[color_switch])
        #axs[i].get_xaxis().set_visible(False)
        #axs[i].axis('off')
    axs[neurons-1].set_xlabel('time (s)')
    return fig, axs

def input_signal(graph, time_plot, neurons_plot, input_pl):
    fig_title_inp = 'Input of ' + graph.title
    fig_inp, ax_inp = plt.subplots()
    ax_inp.set(ylabel='Input', xlabel='time (s)', title=fig_title_inp)
    #my_cycler = rotating_cycler(graph.n_c)
    #ax_inp.set_prop_cycle(my_cycler)
    index_list = neurons_to_clique(graph)
    ax_inp.set_xlim(time_plot[-5000], time_plot[-1])
    for i in range(neurons_plot):
        label = i if i < graph.n_c else '_nolegend_'
        line, = ax_inp.plot(time_plot, input_pl[i].T, label=label, color=COLORS[index_list[i]])
    ax_inp.legend(frameon=False)
    return fig_inp, ax_inp

def response(graph, ext_weights, ax=None, plot_init=False):
    n_clique = graph.n_c
    if ax is None:
        fig, ax = plt.subplots()
    resp, most_responding, second_most_responding = semantic.clique_responses(graph, ext_weights)
    #plot_init = False
    #if initial_weights is not None:
    #    init_resp, init_most = semantic.clique_responses(graph, initial_weights)
    #    plot_init = True
    for i in range(n_clique):
        #pdb.set_trace()
        line, = ax.plot(resp[i], label='C {}'.format(i), color=COLORS[i])
        if plot_init:
            line.set(alpha=0.5, ls='--')

    #ax.legend(frameon=False)
    #response_formula = r'$ R(C, P) = \frac{1}{S_C} \sum_{i \in C, \, j} v_{ij} y_j^P $'
    ax.set_title('Clique response R to pattern P')
    #plt.title(r'$ R(C, P) = \frac{1}{S_C} \sum_{i \in C \, j} v_{ij} y_j^P $')
    ax.set(ylabel='Response R')
    n_pattern = int(np.sqrt(ext_weights.shape[1])) * 2
    ax.set_xticks(np.arange(n_pattern))
    ax.set_xticklabels([])
    #pdb.set_trace()
    return ax, most_responding, second_most_responding

def patterns(graph, n_pattern, axs=None, given_bars=None):
    if axs is None:
        fig, axs = plt.subplots(1, n_pattern)
    bar_size = graph.n_c // 2
    for i, ax in enumerate(axs):
        if given_bars is None:
            matrix = semantic.a_bar(bar_size, [i], False)
            #ax.matshow(matrix)#, cmap='viridis')
        else:
            #pattern, num_bars = dynamics.bars_input(bar_size, 1/bar_size)
            #matrix = np.reshape(pattern, (bar_size, bar_size))
            #pdb.set_trace()
            matrix = semantic.a_bar(bar_size, given_bars[i], False)
        ax.matshow(matrix, cmap='Greys')
        ax.set_xticks(np.arange(-.5, bar_size, 1))
        ax.set_yticks(np.arange(-.5, bar_size, 1))
        ax.grid(lw=1, c='k')
        ax.tick_params(axis='both', left=False, top=False, right=False,
                       bottom=False, labelleft=False, labeltop=False,
                       labelright=False, labelbottom=False)

        #axs[i].set(xticks = [], yticks = [])
    return axs

def receptive_fields(graph, ext_weights, axs=None, ordering=None):
    rec_fields, Min, Max = semantic.receptive_fields(graph, ext_weights)
    if axs is None:
        fig, axs = plt.subplots(1, graph.n_c)

    if ordering is None:
        loop_over = range(len(axs))
    else:
        loop_over = ordering
    for i, j in enumerate(loop_over):
        img = axs[i].matshow(rec_fields[j], vmin=Min, vmax=Max)
        #img = axs[i].matshow(rec_fields[j], vmin=0, vmax=1)
        axs[i].tick_params(axis='both', left=False, top=False, right=False,
                           bottom=False, labelleft=False, labeltop=False,
                           labelright=False, labelbottom=False)
        # cycles over top, bottom, left, right
        for spine in axs[i].spines.keys():
            axs[i].spines[spine].set(color=COLORS[j], lw=2)

    #tick_min = round(Min, -int(floor(log10(abs(Min)))))
    #tick_max = round(Max, -int(floor(log10(abs(Max)))))
    #fig.colorbar(img, ax=axs, ticks=[tick_min, tick_max], aspect=5)
    return axs, img

def stat_receptive_fields(fields, deviation):
    bars_num = fields.shape[0]
    bar_size = fields.shape[1]
    fig, axs = plt.subplots(2, bars_num)
    vmax = fields.max()
    for i, row in enumerate(axs):
        if i == 0:
            for j, field in enumerate(fields):
                img = row[j].matshow(field, vmin=0, vmax=vmax)
                pattern = semantic.a_bar(bar_size, [j], ravel=False)
                row[j].set_xlabel('{:3.2f}'.format((field * pattern).sum()))
                row[j].tick_params(axis='both', left=False, top=False, right=False,
                   bottom=False, labelleft=False, labeltop=False,
                   labelright=False, labelbottom=False)
        else:
            for j, dev in enumerate(deviation):
                img = row[j].matshow(dev, vmin=0, vmax=vmax)
                pattern = semantic.a_bar(bar_size, [j], ravel=False)
                row[j].set_xlabel('{:3.2f}'.format((dev * pattern).sum()))
                row[j].tick_params(axis='both', left=False, top=False, right=False,
                   bottom=False, labelleft=False, labeltop=False,
                   labelright=False, labelbottom=False)
    return fig, axs, img

def complete_figure(graph, ext_weights, initial_weights=None):
    ''' draws a figure with clique response, patterns, and receptive fields'''
    fig_compl = plt.figure()
    grid_rows = 4
    n_pattern = int(np.sqrt(ext_weights.shape[1])) * 2
    grid_spec = gridspec.GridSpec(grid_rows, graph.n_c + 1, 
                height_ratios=[8, 1, 1, 1], width_ratios=[*[1]*graph.n_c, 0.2])
    ax_resp = plt.subplot(grid_spec[0, :-1])
    ax_resp, most_responding, second_most_responding = response(graph, ext_weights, ax=ax_resp)
    if initial_weights is not None:
        response(graph, initial_weights, ax=ax_resp, plot_init=True)

    axs_bars = []
    axs_recep_max = []
    axs_recep = []
    for i in range(n_pattern):
        axs_bars.append(plt.subplot(grid_spec[1, i]))
        axs_recep_max.append(plt.subplot(grid_spec[2, i]))
    for i in range(graph.n_c):    
        axs_recep.append(plt.subplot(grid_spec[3, i]))

    ax_colorbar = plt.subplot(grid_spec[2:, -1])
    
    axs_bars = patterns(graph, n_pattern, axs_bars)
    axs_recep_max, img = receptive_fields(graph, ext_weights, axs_recep_max, most_responding)
    axs_recep, img = receptive_fields(graph, ext_weights, axs_recep, second_most_responding)
    plt.colorbar(img, cax=ax_colorbar)
    axs_recep_max[0].set_ylabel('Largest\nresponse\n')
    axs_recep[0].set_ylabel('Second\nlargest\n')
    axs_recep_max[3].set_title('       Receptive Fields')
    fig_compl.set_size_inches([7.0, 13.])
    plt.subplots_adjust(left=0.13, right=1-0.13, top=0.95, bottom=0.05, hspace=0.05)
    plt.savefig('./log/double_complete_tall.pdf', dpi=300)
    return fig_compl, ax_resp, axs_bars, axs_recep_max, axs_recep, ax_colorbar
