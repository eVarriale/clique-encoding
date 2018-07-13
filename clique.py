#!/usr/bin/env python3

import numpy as np                  # for math
import matplotlib.pyplot as plt     # for plots
from matplotlib.ticker import FormatStrFormatter
plt.ion()
import pdb                          # for debugging
from tqdm import tqdm               # for progress bar
#pdb.set_trace()
from scipy.stats import norm        # for distributions
import scipy.linalg as LA           # for linear algebra
import networkx as nx               # for graphs
from cycler import cycler           # for managing plot colors

import time

import networks as net
import dynamics
import plotting as myplot

# Global variables

SIM_STEPS       = 1*60*1000      # integration steps 
DELTA_T         = 1             # integration time step in milliseconds

PROB_CONN   = 0.3
EXC_FRAC    = 0.8
np.random.seed(5)

w_mean          = 1


n_clique = 3 # >=3 ! (for s_c = 3)
clique_size = 5 # >2 !, otherwise cliques don't make sense
NEURONS = n_clique * clique_size
sparseness = True

gain_rule = 2 # 1 : set target variance, 2 : self organized variance, 3 : gain is fixed
if gain_rule == 3:
    subtitle = 'Target variance = {:.2f}'.format(TARGET_VAR)
elif gain_rule == 2:
    subtitle = 'Self-organized variance'
else:
    subtitle = 'Gain is fixed'
print(subtitle)

if sparseness:
    if n_clique <= (2 * clique_size + 3)/clique_size:
        neurons = n_clique * clique_size
        exc_links = neurons * (clique_size +1) // 2
        tot_links = neurons*(neurons-1)//2
        p_inh = exc_links/(tot_links - exc_links)
        print('All-to-all even if sparse! P_inh = {}'.format(p_inh))



#weights, G = geometric_network(n_clique, clique_size, sparse, w_mean)
PROB_CONN = 0.3
#weights, G, G_exc = ErdosRenyi_network()

weights, G = net.rotating_clique(n_clique, clique_size, sparseness, w_mean)
'''
def plot_network(G):
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
    ax_net.set_title('Network of {} cliques with {} nodes, sparse = {}'.format(n_clique, clique_size, sparseness))
    ax_net.axis('off')
'''
if NEURONS < 20:
    myplot.network(G)

#first_clique = np.ones(clique_size)
#x = np.concatenate((first_clique, np.random.rand(NEURONS - clique_size)))
x = np.random.rand(NEURONS)
gain = np.random.normal(3, 0.5, NEURONS)
threshold = np.random.normal(0.5, 0.1, NEURONS)

Y = np.zeros((NEURONS, SIM_STEPS))
inputs = np.zeros((NEURONS, SIM_STEPS))
B = np.zeros((NEURONS, SIM_STEPS))
A = np.zeros((NEURONS, SIM_STEPS))
X = np.zeros((NEURONS, SIM_STEPS))

for t in tqdm(range(SIM_STEPS)):
    #x, threshold, gain, activity, x_inp = dynamics.target(x, gain, threshold, weights, gain_rule)
    dx, db, da, activity, x_inp = dynamics.target(x, gain, threshold, weights, gain_rule)
    x += dx * DELTA_T
    threshold += db * DELTA_T
    gain += da * DELTA_T
    Y[:, t] = activity
    B[:, t] = threshold
    A[:, t] = gain
    inputs[:, t] = x_inp
    X[:, t] = x

#final_times = 10000
#final_times = int(SIM_STEPS/3)
final_times = SIM_STEPS
time_plot = np.arange(final_times)/int(1000/DELTA_T) # time in seconds
neurons_plot = NEURONS
Y_plot = Y[:, -final_times:]
mean = Y_plot.mean()
std = Y_plot.std()
if NEURONS < 20 and False:
    fig, axs = plt.subplots(NEURONS, 1, sharex=True)
    plt.suptitle(fig_title)
    plt.title(subtitle)
    fig.subplots_adjust(hspace=0)
    color = { 0 : 'xkcd:pale blue', 1 : 'xkcd:white'}
    color_switch = True
    for i in range(NEURONS):
        if(i % clique_size==0):
            color_switch = not color_switch
        axs[i].set_ylim(-1.1, 2.1)
        axs[i].plot(time_plot, Y_plot[i], 'k', linewidth=1, label = i%n_clique)
        axs[i].fill_between(time_plot, 0, Y_plot[i])
        axs[i].plot(time_plot, B[i, -final_times:], 'r--', linewidth=1)
        axs[i].set_facecolor(color[color_switch])
        #axs[i].get_xaxis().set_visible(False)
        #axs[i].axis('off')
    axs[NEURONS-1].set_xlabel('time (s)'.format((SIM_STEPS-final_times)/int(1000/DELTA_T)))


input_pl = inputs[:, -final_times:]
X_plot = X[:, -final_times:]

'''
fig_inp, axs_inp = plt.subplots(NEURONS, 1, sharex=True)
plt.suptitle('Input of {} cliques with {} nodes'.format(n_clique, clique_size))
plt.title(subtitle)
fig_inp.subplots_adjust(hspace=0)
color_switch = True
for i in range(NEURONS):
    if(i % clique_size==0):
        color_switch = not color_switch
    #axs_inp[i].set_ylim(-1, 2)
    axs_inp[i].plot(time_plot, input_pl[i]-X_plot[i], 'k', linewidth=1)
    #axs_inp[i].fill_between(time_plot, 0, input_pl[i])
    #plt.yticks([0.5], ('{}'.format(i)))
    axs_inp[i].set_facecolor(color[color_switch])
    #axs_inp[i].get_xaxis().set_visible(False)
    #axs_inp[i].axis('off')
axs_inp[NEURONS-1].set_xlabel('time (s)')
'''

save_figures = True
fig_title = 'Activity of {} cliques with {} nodes, sparse = {}'.format(G.n_c, G.s_c, G.sparse)

myplot.activity(G, time_plot, neurons_plot, Y_plot, A, B, gain_rule, save_figures)
myplot.input(G, time_plot, neurons_plot, input_pl, save_figures)

'''
fig2, ax2 = plt.subplots()
ax2.hist(Y_plot.flatten(), density=True, bins = 50)
hist_x = np.linspace(0, 1, 1000)
ax2.plot(hist_x, norm.pdf(hist_x, loc = TARGET_MEAN, scale = TARGET_VAR**0.5), label = 'Target: $\mathcal{N}$'+'({}, {:.1})'.format(TARGET_MEAN, np.sqrt(TARGET_VAR)))
ax2.plot(hist_x, norm.pdf(hist_x, loc = mean, scale = std), label = '$\mathcal{N}'+'(\mu_\mathrm{y}, \, \sigma_\mathrm{y})$')
#ax2.hist(Y[:N_EXC, :].flatten(), density=True, bins = 100, histtype = 'step', label = 'excitatory')
#ax2.hist(Y[N_EXC:,:].flatten(), density=True, bins = 100, histtype = 'step', label = 'excitatory')
ax2.set(title = 'Activity distribution, exc: {}, w_mean: {}'.format(EXC_FRAC, w_mean), xlabel = 'Activity y', ylabel = 'Density')
ax2.legend()
if save_figures:
    fig2.savefig('exc{}_w_mean{}_distr'.format(int(EXC_FRAC*10), int(w_mean*100)), **savefig_options)
    plt.close(fig2)
    #Y_plot = 0    

fig3, ax3 = plt.subplots()
ax3.plot(time_plot, B[:neurons_plot, -final_times:].T)
ax3.set(ylabel = 'Threshold', xlabel = 'time (s)', title = 'excitatory fraction: {}, w_mean: {}'.format(EXC_FRAC, w_mean))
if save_figures:
    fig3.savefig('exc{}_w_mean{}_thresh'.format(int(EXC_FRAC*10), int(w_mean*100)), **savefig_options)
    plt.close(fig3)
    #B = 0

fig4, ax4 = plt.subplots()
ax4.plot(time_plot, A[:neurons_plot, -final_times:].T)
ax4.set(ylabel = 'Gain', xlabel = 'time (s)', title = 'excitatory fraction: {}, w_mean: {}'.format(EXC_FRAC, w_mean)) 
if save_figures:
    fig4.savefig('exc{}_w_mean{}_gain'.format(int(EXC_FRAC*10), int(w_mean*100)), **savefig_options)
    plt.close(fig4)
    #A = 0
'''
'''
fig5, ax5 = plt.subplots()
ax5.hist(inputs[:, -final_times:].flatten(), density=True, bins = 50)
ax5.set(title = 'Input distribution, exc: {}, w_mean: {}'.format(EXC_FRAC, w_mean), xlabel = 'Input', ylabel = 'Density') 
if save_figures:
    fig5.savefig('exc{}_w_mean{}_inpdistr'.format(int(EXC_FRAC*10), int(w_mean*100)), **savefig_options)
    plt.close(fig5)
    #inputs = 0
'''
plt.show()