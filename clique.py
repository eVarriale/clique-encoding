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

sim_steps = 1*10*1000 # integration steps
delta_t = 1 # integration time step in milliseconds

w_mean = 1

n_clique = 8
# must be even because of bars problem
clique_size = 1 # >2 !, otherwise cliques don't make sense


gain_rule = 0 # 1 : set target variance, 2 : self organized variance, 3 : gain is fixed

input_on = sim_steps
avg_bars = 1 # average # of bars per input

T_patt = 40 * 10# * 5# / delta_t #30  # bars lasting time
T_inter = 100 * 10# / delta_t  # time beetween patterns

v_jl_sampling = 1000
bars_start = sim_steps# * 0
seed = 5

# Initialize stuff

np.random.seed(seed)

if gain_rule == 1:
    subtitle = 'Target variance' # = {:.2f}'.format(TARGET_VAR)
elif gain_rule == 2:
    subtitle = 'Self-organized variance'
elif gain_rule == 3:
    subtitle = 'Gain is fixed'
elif gain_rule == 0:
    subtitle = 'Full depletion model'
print(subtitle)

#weights, graph, G_exc = ErdosRenyi_network()

#w_jk, z_jk, graph = net.rotating_clique_ring(n_clique, clique_size)
#w_jk, z_jk, graph = net.rotating_clique_net(n_clique) # prof
w_jk, z_jk, graph = net.geometric_net(n_clique)
#w_jk, z_jk, graph = net.geometric_ring(n_clique, clique_size)
#w_jk, z_jk, graph = net.ring(n_clique, w_exc=0.4, w_inh=1)
weights = w_jk + z_jk
neurons = graph.number_of_nodes()


plot_network = False
if neurons < 20 and plot_network:
    fig_net, ax_net = myplot.network(graph)

x = np.random.rand(neurons)
#x = np.array([-0.391]*neurons)
#first_clique = np.ones(clique_size)
#x[graph.clique_list[0]] = 10.

neurons_over_time = (neurons, sim_steps)
activity_record = np.zeros(neurons_over_time)
input_record = np.zeros(neurons_over_time)
sensory_inp_record = np.zeros(neurons_over_time)
membrane_pot_record = np.zeros(neurons_over_time)


a = np.random.normal(9, 0.1, neurons)
b = np.random.normal(0.5, 0.1, neurons)
gain_record = np.zeros(neurons_over_time)
threshold_record = np.zeros(neurons_over_time)

φ_inh = np.random.normal(1, 0.1, neurons)
u_inh = np.random.normal(1, 0.1, neurons)
full_vesicles_inh_record = np.zeros(neurons_over_time)
vesic_release_inh_record = np.zeros(neurons_over_time)

φ_exc = np.random.normal(1, 0.1, neurons)
u_exc = np.random.normal(1, 0.1, neurons)
full_vesicles_exc_record = np.zeros(neurons_over_time)
vesic_release_exc_record = np.zeros(neurons_over_time)

how_many_inputs = 0
how_many_bars = 0

bar_size = n_clique // 2
if bar_size == 0:
    bar_size = 1
input_size = bar_size**2

p_bars = np.minimum(avg_bars / n_clique, 1)

v_jl = net.external_weights(input_size, neurons, p_bars)
v_jl_0 = copy.copy(v_jl)

sensory_weight_record = np.zeros((neurons, input_size, sim_steps//v_jl_sampling))
ext_signal = np.zeros(input_size)

bars_time = np.zeros(sim_steps) - 1

learn_record = np.zeros((neurons, input_size, sim_steps))
dec_record = np.zeros((neurons, input_size, sim_steps))


# Main simulation

for time in tqdm(range(sim_steps)):

    # the sensory signal is activated every T_inter ms, for T_patt ms
    if time >= bars_start:
        if time == input_on + T_patt:
            ext_signal *= 0 
            if neurons == 1 : ext_signal = np.array([-10])
            bars_time[input_on:time] = bars_num
        if time%T_inter == 0: 
            ext_signal, bars_num = dynamics.bars_input(bar_size, p_bars)
            if neurons == 1 : ext_signal = np.array([10])
            input_on = time
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
    variation = dynamics.full_depletion(x, φ_inh, u_inh, w_jk,
                                        z_jk, ext_signal, v_jl)
    dx, du_i, dφ_i, y, T, S, dV, learn, dec = variation
    φ_inh += dφ_i
    u_inh += du_i
    full_vesicles_inh_record[:, time] = φ_inh
    vesic_release_inh_record[:, time] = u_inh
    #φ_exc += dφ_e
    #u_exc += du_e
    #full_vesicles_exc_record[:, time] = φ_exc
    #vesic_release_exc_record[:, time] = u_exc    

    x += dx * delta_t
    if not time % (sim_steps//100) and np.isnan(x).any(): 
        print('\nNaN detected!\n')
        break
    v_jl += dV * delta_t
    
    activity_record[:, time] = y# * (dV[:, 3] < 0).any()
    membrane_pot_record[:, time] = x
    input_record[:, time] = T
    sensory_inp_record[:, time] = S
    if time%v_jl_sampling == 0:
        sensory_weight_record[:, :, time//v_jl_sampling] = v_jl

    learn_record[:, :, time] = learn
    dec_record[:, :, time] = dec
    


# Plotting

# Setting up...

#final_times = 10000
#final_times = int(sim_steps/3)
final_times = np.minimum(sim_steps, 10000)
time_plot = np.arange(sim_steps-final_times, sim_steps)/int(1000/delta_t) # time in seconds
neurons_plot = neurons
y_plot = activity_record[:, -final_times:]
input_pl = input_record[:, -final_times:]
x_plot = membrane_pot_record[:, -final_times:]
gain_plot = gain_record[:, -final_times:]
threshold_plot = threshold_record[:, -final_times:]
sens_inp_plot = sensory_inp_record[:, -final_times:]
full_vesicles_inh_plot = full_vesicles_inh_record[:, -final_times:]
vesic_release_inh_plot = vesic_release_inh_record[:, -final_times:]

learn_plot = learn_record[:, :, -final_times:]
dec_plot = dec_record[:, :, -final_times:]

# Actually plotting stuff

list_of_plots = {}
if not (vesic_release_inh_record==0).all() and neurons == 1:
    fig_fulldep, ax_fulldep = plt.subplots()
    plt.plot(time_plot, full_vesicles_inh_record.T, label = 'φ')
    plt.plot(time_plot, vesic_release_inh_record.T, label = 'u')
    effective_weights_records = vesic_release_inh_record * full_vesicles_inh_record
    plt.plot(time_plot, effective_weights_records.T, label = 'φ$\cdot $u')
    ax_fulldep.set(ylim=[-0.02, dynamics.U_max + .02], xlim=[1.8, 2.8])
    ax_fulldep.set(xlabel='time (s)')
    ax_fulldep.set_yticks([0, 1, effective_weights_records.max(), dynamics.U_max])
    ax_fulldep.set_yticklabels(['0', '1', '{:2.1f}'.format(effective_weights_records.max()), '$U_{max}$'])
    #ax_fulldep.set_xticks([1.8, 2.8])
    ax_fulldep.set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])
    plt.legend(frameon=False, prop={'size': 15})
    plt.tight_layout()
    list_of_plots['full_depletion'] = fig_fulldep

save_figures = False
effect_plot = vesic_release_inh_plot * full_vesicles_inh_plot
fig_ac, ax_ac = myplot.activity(graph, time_plot, neurons_plot, y_plot,
                                threshold_plot, learn_plot[:,0,:], 0
                                 ,bars_time=bars_time)
list_of_plots['activity'] = fig_ac

if (v_jl != v_jl_0).any() and neurons > 1:
    fig_output = myplot.complete_figure(graph, v_jl, v_jl_0)
    fig_compl, ax_resp, axs_bars, axs_recep = fig_output
    fig_compl.set_size_inches([7.01, 9.51])
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
plt.show()

def save_stuff():
    version = 0
    for key, figure in list_of_plots.items():
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                file_name = './log/'+str(version)+'_'+key+'.pdf'
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