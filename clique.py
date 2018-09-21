#!/usr/bin/env python3

import copy
import time
import importlib
import pdb                          # for debugging
import numpy as np                  # for math
import matplotlib.pyplot as plt     # for plots
from tqdm import tqdm               # for progress bar
#pdb.set_trace()
#from scipy.stats import norm        # for distributions
#import scipy.linalg as LA           # for linear algebra
#import networkx as nx               # for graphs
#from cycler import cycler           # for managing plot colors
import matplotlib

#plt.style.use('seaborn-poster')
#plt.style.use('seaborn-talk')
#plt.style.use('seaborn-white')
import networks as net
import dynamics
import plotting as myplot
import semantic

plt.ion()

def reload_mod():
    importlib.reload(net)
    importlib.reload(dynamics)
    importlib.reload(myplot)
    importlib.reload(semantic)

# Global variables

sim_steps = 1 * 60 * 1000 # integration steps
delta_t = 1 # integration time step in milliseconds
trials_num = 1
w_mean = 1

n_clique = 8
# must be even because of bars problem
clique_size = 7 # >2 !, otherwise cliques don't make sense


gain_rule = 0 # 1 : set target variance, 2 : self organized variance, 3 : gain is fixed

avg_bars = 1 # average # of bars per input

T_patt = 30 * (1 + 19 * (n_clique == 1))# * 5# / delta_t #30  # bars lasting time
T_inter = 100 * (1 + 19 * (n_clique == 1))# / delta_t  # time beetween patterns

v_jl_sampling = 1000
bars_start = sim_steps# * 0
seed = 6

# Initialize stuff

np.random.seed(seed)

#weights, graph, G_exc = ErdosRenyi_network()

#w_jk, z_jk, graph = net.rotating_clique_ring(n_clique, clique_size)
#w_jk, z_jk, graph = net.rotating_clique_net(n_clique) # prof
w_jk, z_jk, graph = net.geometric_net(n_clique) # FLOWER
if n_clique == 1:
    w_jk, z_jk, graph = net.geometric_ring(n_clique, clique_size) 
    # use this for 1 neuron
#w_jk, z_jk, graph = net.ring(n_clique, w_exc=0.4, w_inh=1)
weights = w_jk + z_jk
neurons = graph.number_of_nodes()
neurons_over_time = (neurons, sim_steps)

bar_size = n_clique // 2
if bar_size == 0:
    bar_size = 1
input_size = bar_size**2

fields_size = (n_clique, bar_size, bar_size)

first_field_avg = np.zeros(fields_size)
second_field_avg = np.zeros(fields_size)
first_field_square = np.zeros(fields_size)
second_field_square = np.zeros(fields_size)

first_response_avg = np.zeros(n_clique)
second_response_avg = np.zeros(n_clique)
first_response_square = np.zeros(n_clique)
second_response_square = np.zeros(n_clique)
        
for trial in tqdm(range(trials_num)):

    #x = np.random.rand(neurons)
    x = np.ones(neurons)
    #first_clique = np.ones(clique_size)
    #x[graph.clique_list[0]] = 10.

    activity_record = np.zeros(neurons_over_time)
    input_record = np.zeros(neurons_over_time)
    sensory_inp_record = np.zeros(neurons_over_time)
    membrane_pot_record = np.zeros(neurons_over_time)


    #a = np.random.normal(9, 0.1, neurons)
    #b = np.random.normal(0.5, 0.1, neurons)
    #gain_record = np.zeros(neurons_over_time)
    #threshold_record = np.zeros(neurons_over_time)

    φ_inh = 1#np.random.normal(1, 0.1, neurons)
    u_inh = 1#np.random.normal(1, 0.1, neurons)
    full_vesicles_inh_record = np.zeros(neurons_over_time)
    vesic_release_inh_record = np.zeros(neurons_over_time)

    φ_exc = 1#np.random.normal(1, 0.1, neurons)
    u_exc = 1#np.random.normal(1, 0.1, neurons)
    full_vesicles_exc_record = np.zeros(neurons_over_time)
    vesic_release_exc_record = np.zeros(neurons_over_time)

    t_input_on = sim_steps
    input_on = False
    how_many_inputs = 0
    how_many_bars = 0


    p_bars = np.minimum(avg_bars / n_clique, 1)

    v_jl = net.external_weights(input_size, neurons, p_bars)
    v_jl_0 = copy.copy(v_jl)

    #sensory_weight_record = np.zeros((neurons, input_size, sim_steps//v_jl_sampling))
    ext_signal = np.zeros(input_size)

    bars_time = np.zeros(sim_steps) - 1

    #learn_record = np.zeros((neurons, input_size, sim_steps))
    #dec_record = np.zeros((neurons, input_size, sim_steps))


    # Main simulation

    for time in tqdm(range(sim_steps)):

        # the sensory signal is activated every T_inter ms, for T_patt ms
        if time >= bars_start:
            if time == t_input_on + T_patt and input_on:
                ext_signal *= 0
                if neurons == 1: 
                    ext_signal = np.array([-10])
                bars_time[t_input_on:time] = bars_num
                input_on = False
            if time%T_inter == 0 and not input_on:
                ext_signal, bars_num = dynamics.bars_input(bar_size, p_bars)
                if neurons == 1: 
                    ext_signal = np.array([10])
                    bars_num = 1
                t_input_on = time
                input_on = True
                how_many_inputs += 1
                how_many_bars += bars_num

        '''
        # dynamics for sliding threshold model
        # x : membrane potential, b : threshold, a : gain
        # y : activity, T : total input, S : sensory input
        dx, db, da, y, T, S, dV = dynamics.target(x, a, b, weights, gain_rule, ext_signal, v_jl)
        b = np.maximum( b + db * delta_t, 0.)
        a += da * delta_t
        threshold_record[:, time] = b
        gain_record[:, time] = a
        '''
        # dynamics for Tsodyks-Markram model
        # x : membrane potential, u_inh : vesicle release factor, φ_inh : number of full vesicles
        # y : activity, T : total input, S : sensory input
        # w_jk : excitatory recurrent, z_jk : inhibitory rec, v_jl : exc sensory
        variation = dynamics.full_depletion(x, φ_inh, u_inh, φ_exc, u_exc, w_jk,
                                            z_jk, ext_signal, v_jl)
        #dx, du_i, dφ_i, y, T, S, dV, learn, dec = variation
        dx, du_i, dφ_i, du_e, dφ_e, y, T, S, dV, learn, dec = variation
        φ_inh += dφ_i
        u_inh += du_i
        full_vesicles_inh_record[:, time] = φ_inh
        vesic_release_inh_record[:, time] = u_inh
        φ_exc += dφ_e
        u_exc += du_e
        full_vesicles_exc_record[:, time] = φ_exc
        vesic_release_exc_record[:, time] = u_exc

        x += dx * delta_t
        if not time % (sim_steps//100) and np.isnan(x).any():
            print('\nNaN detected!\n')
            break
        v_jl += dV * delta_t

        activity_record[:, time] = y# * (dV[:, 3] < 0).any()
        membrane_pot_record[:, time] = x
        input_record[:, time] = T
        sensory_inp_record[:, time] = S
        #if time%v_jl_sampling == 0:
        #    sensory_weight_record[:, :, time//v_jl_sampling] = v_jl

        #learn_record[:, :, time] = learn
        #dec_record[:, :, time] = dec

    # End of ONE simulation
    fields_1st, fields_2nd, resp_1st, resp_2nd = semantic.best_receptive_fields(graph, v_jl)
    
    first_field_avg += fields_1st
    second_field_avg += fields_2nd
    first_field_square += fields_1st**2
    second_field_square += fields_2nd**2

    first_response_avg += resp_1st
    second_response_avg += resp_2nd
    first_response_square += resp_1st**2
    second_response_square += resp_2nd**2

# End of ALL simulations
first_field_avg /= trials_num
second_field_avg /= trials_num
first_field_square /= trials_num
second_field_square /= trials_num


first_field_sdv = np.sqrt((first_field_square - first_field_avg**2)/trials_num)
second_field_sdv = np.sqrt((second_field_square - second_field_avg**2)/trials_num)

first_response_avg /= trials_num
second_response_avg /= trials_num
first_response_square /= trials_num
second_response_square /= trials_num

first_response_dev = np.sqrt((first_response_square - first_response_avg**2)/trials_num)
second_response_dev = np.sqrt((second_response_square - second_response_avg**2)/trials_num)

# Plotting

# Setting up...

#final_times = 30000
#final_times = sim_steps
final_times = np.minimum(sim_steps, 10000)
time_plot = np.arange(sim_steps-final_times, sim_steps)/int(1000/delta_t) # time in seconds
neurons_plot = neurons
y_plot = activity_record[:, -final_times:]
input_pl = input_record[:, -final_times:]

x_plot = membrane_pot_record[:, -final_times:]
#gain_plot = gain_record[:, -final_times:]
#threshold_plot = threshold_record[:, -final_times:]
sens_inp_plot = sensory_inp_record[:, -final_times:]
full_vesicles_inh_plot = full_vesicles_inh_record[:, -final_times:]
vesic_release_inh_plot = vesic_release_inh_record[:, -final_times:]
effective_weights_plot_inh = vesic_release_inh_plot * full_vesicles_inh_plot

full_vesicles_exc_plot = full_vesicles_exc_record[:, -final_times:]
vesic_release_exc_plot = vesic_release_exc_record[:, -final_times:]
effective_weights_plot_exc = vesic_release_exc_plot * full_vesicles_exc_plot

#learn_plot = learn_record[:, :, -final_times:]
#dec_plot = dec_record[:, :, -final_times:]

# Actually plotting stuff

list_of_plots = {}
if neurons == 1 and not (vesic_release_inh_record == 0).all():
    fig_stp, ax_stp = myplot.full_depletion(time_plot, full_vesicles_inh_plot,
                                            vesic_release_inh_plot,
                                            full_vesicles_exc_plot,
                                            vesic_release_exc_plot)
    #plt.savefig('./notes/Poster/hendrik/images/double_depletion.pdf', dpi=300)
    list_of_plots['full_depletion'] = fig_fulldep

save_figures = False

fig_ac, ax_ac = myplot.activity(graph, time_plot, neurons_plot, y_plot, effective_weights_plot_exc,
                                -effective_weights_plot_inh, 1
                                , bars_time=bars_time)

#ax_ac.set_xlim([0,10])
#plt.savefig('./notes/Poster/hendrik/images/double_activity.pdf', dpi=300)                              
#list_of_plots['activity'] = fig_ac

if (v_jl != v_jl_0).any() and neurons > 1:
    fig_output = myplot.complete_figure(graph, v_jl, v_jl_0)
    fig_compl, ax_resp, axs_bars, axs_recep_max, axs_recep, ax_clrbar = fig_output
    
    fig_compl.suptitle('T_f = {} T_v = {}'.format(dynamics.T_f, dynamics.T_v))
    list_of_plots['complete'] = fig_compl
'''
x_der = (x_plot[:, 1:] - x_plot[:, :-1]) / delta_t
fig_der_x, ax_der_x = myplot.activity(graph, time_plot[:-1], neurons_plot, x_der,
                                  gain_record[:, :-1], threshold_record[:, :-1],
                                  gain_rule, save_figures)

fig_x, ax_x = myplot.activity(graph, time_plot[:-1], neurons_plot, x_plot[:, :-1],
                                  gain_record[:, :-1], threshold_record[:, :-1],
                                  gain_rule, save_figures)

y_der = (y_plot[:, 1:] - y_plot[:, :-1]) / delta_t
y_2 = dynamics.activation(x_plot[:,:-1], dynamics.gain)
y_der2 = dynamics.gain * y_2 * (1 - y_2) * x_der
fig_der_y, ax_der_y = myplot.activity(graph, time_plot[:-1], neurons_plot, y_der,
                                  gain_record[:, :-1], threshold_record[:, :-1],
                                  gain_rule, save_figures)
'''
#myplot.input_signal(graph, time_plot, neurons_plot, input_pl)
if trials_num > 1:
    fig_rec1, axs_rec1, img1 = myplot.stat_receptive_fields(first_field_avg, first_field_sdv)
    fig_rec1.colorbar(img1, ax=fig_rec1.axes)
    fig_rec1.suptitle('Best receptive fields')

    fig_rec2, axs_rec2, img2 = myplot.stat_receptive_fields(second_field_avg, second_field_sdv)
    fig_rec2.colorbar(img2, ax=fig_rec2.axes)
    fig_rec2.suptitle('Second best receptive fields')

    fig_resp_stat, ax_resp_stat = plt.subplots()
    ax_resp_stat.errorbar(np.arange(n_clique), first_response_avg, yerr=first_response_dev, label='First response')
    lines = ax_resp_stat.errorbar(np.arange(n_clique), second_response_avg, yerr=second_response_dev, label='Second response')
    #lines[-1][0].set_linestyle('--')
    plt.legend()
plt.show()

def save_stuff():
    version = 0
    for key, figure in list_of_plots.items():
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                file_name = './log/' + str(version) + '_' + key + '.pdf'
                #print(file_name)
                f = open(file_name, 'x')
            except FileExistsError as error:
                version += 1
            else:
                break
        if version == max_attempts:
            print('Too many saved files')
        else:
            figure.savefig(file_name, dpi=300)
        version = 0
