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

prob_conn = 0.3
np.random.seed(5)

w_mean = 1

n_clique = 10 # >=3 ! (for s_c = 3)
# must be even because of bars problem
#if n_clique % 2:
#    n_clique += 1
clique_size = 10 # >2 !, otherwise cliques don't make sense
neurons = n_clique * clique_size
sparseness = False

gain_rule = 0 # 1 : set target variance, 2 : self organized variance, 3 : gain is fixed

if gain_rule == 1:
    subtitle = 'Target variance' # = {:.2f}'.format(TARGET_VAR)
elif gain_rule == 2:
    subtitle = 'Self-organized variance'
elif gain_rule == 3:
    subtitle = 'Gain is fixed'
elif gain_rule == 0:
    subtitle = 'Full depletion model'
print(subtitle)

if sparseness:
    if n_clique <= (2 * clique_size + 3)/clique_size:
        exc_links = neurons * (clique_size +1) // 2
        tot_links = neurons*(neurons-1)//2
        p_inh = exc_links/(tot_links - exc_links)
        print('All-to-all even if sparse! P_inh = {}'.format(p_inh))

#weights, graph, G_exc = ErdosRenyi_network()

w_jk, z_jk, graph = net.rotating_clique(n_clique, clique_size, sparseness, w_mean)
#w_jk, z_jk, graph = net.geometric(n_clique, clique_size)
#w_jk, z_jk, graph = net.ring(n_clique, w_exc=0.4, w_inh=1)
weights = w_jk + z_jk

plot_network = False
if neurons < 20 and plot_network:
    fig_net, ax_net = myplot.network(graph)

x = np.random.rand(neurons)
#x = np.array([-0.391]*neurons)
#first_clique = np.ones(clique_size)
#x[graph.clique_list[0]] = 10.

activity_record = np.zeros((neurons, sim_steps))
input_record = np.zeros((neurons, sim_steps))
sensory_inp_record = np.zeros((neurons, sim_steps))
membrane_pot_record = np.zeros((neurons, sim_steps))


a = np.random.normal(9, 0.1, neurons)
b = np.random.normal(0.5, 0.1, neurons)
gain_record = np.zeros((neurons, sim_steps))
threshold_record = np.zeros((neurons, sim_steps))

φ = np.random.normal(1, 0.1, neurons)
u = np.random.normal(1, 0.1, neurons)
full_vesicles_record = np.zeros((neurons, sim_steps))
vesic_release_record = np.zeros((neurons, sim_steps))

input_on = 0
bar_size = n_clique // 2
if bar_size == 0:
    bar_size = 1
input_size = bar_size**2
P_bars = np.minimum(2 / n_clique, 1) # average of 2 bars per input

T_patt = 40 / delta_t #30  # bars lasting time
T_inter = 100 / delta_t  # time beetween patterns

v_jl = net.external_weights(input_size, neurons, P_bars)
v_jl_0 = copy.copy(v_jl)
v_jl_sampling = int(1000/delta_t)
sensory_weight_record = np.zeros((neurons, input_size, sim_steps//v_jl_sampling))
ext_signal = np.zeros(input_size)

bars_start = sim_steps
for time in tqdm(range(sim_steps)):

    # the sensory signal is activated every T_inter ms, for T_patt ms
    if time > bars_start:
        if time == input_on + T_patt:
            ext_signal *= 0 
            if neurons == 1 : ext_signal = np.array([-10])
        if time%T_inter == 0: 
            ext_signal = dynamics.bars_input(bar_size, P_bars)
            if neurons == 1 : ext_signal = np.array([10])
            input_on = time

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
    # x : membrane potential, u : vesicle release factor, φ : number of full vesicles
    # y : activity, T : total input, S : sensory input
    # w_jk : excitatory recurrent, z_jk : inhibitory rec, v_jl : exc sensory
    dx, du, dφ, y, T, S, dV = dynamics.full_depletion(x, φ, u, w_jk, z_jk, ext_signal, v_jl)
    φ += dφ
    u += du
    full_vesicles_record[:, time] = φ
    vesic_release_record[:, time] = u
  

    x += dx * delta_t
    v_jl += dV * delta_t
    
    activity_record[:, time] = y# * (dV[:, 3] < 0).any()
    membrane_pot_record[:, time] = x
    input_record[:, time] = T
    sensory_inp_record[:, time] = S
    if time%v_jl_sampling == 0:
        sensory_weight_record[:, :, time//v_jl_sampling] = v_jl


#final_times = 10000
#final_times = int(sim_steps/3)
final_times = sim_steps#np.minimum(sim_steps, 10000)
time_plot = np.arange(sim_steps-final_times, sim_steps)/int(1000/delta_t) # time in seconds
neurons_plot = neurons
Y_plot = activity_record[:, -final_times:]

if not (vesic_release_record==0).all() and False:
    fig, ax = plt.subplots()
    #plt.plot(time_plot, full_vesicles_record.T, label = 'φ')
    #plt.plot(time_plot, vesic_release_record.T, label = 'u')
    plt.plot(time_plot, vesic_release_record.T * full_vesicles_record.T, label = 'φu')
    ax.set(ylim=[0, dynamics.U_max + .01])
    plt.legend()

input_pl = input_record[:, -final_times:]
x_plot = membrane_pot_record[:, -final_times:]
gain_plot = gain_record[:, -final_times:]
threshold_plot = threshold_record[:, -final_times:]
sens_inp_plot = sensory_inp_record[:, -final_times:]

save_figures = False
#Y_plot *= vesic_release_record * full_vesicles_record
fig_ac, ax_ac = myplot.activity(graph, time_plot, neurons_plot, Y_plot, gain_plot, threshold_plot, gain_rule, save_figures)

if (v_jl != v_jl_0).any():
    fig_compl, ax_resp, axs_bars, axs_recep = myplot.complete_figure(graph, v_jl, v_jl_0)
#derivative = ( (Y_plot[:, 1:]-Y_plot[:, :-1])/(time_plot[1] - time_plot[0]) )
#myplot.activity(graph, time_plot[:-1], neurons_plot, derivative, gain_record[:-1], threshold_record[:-1], gain_rule, save_figures)

#myplot.input_signal(graph, time_plot, neurons_plot, input_pl)

plt.show()