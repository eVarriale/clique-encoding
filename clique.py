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

# Global variables

PROB_CONN   = 0.3
EXC_FRAC    = 0.8
W_MEAN      = 1
np.random.seed(4)

DELTA_T         = 1             # integration time step in milliseconds
SIM_STEPS       = 5*60*1000      # integration steps 
GAMMA           = 1./20         # leaking rate
EPS_W           = 1./100000.    # learning rate
EPS_A           = 1./10000.     # gain change rate
EPS_B           = 1./10000.     # threshold change rate


TARGET_MEAN     = 0.3
TARGET_VAR      = 0.1**2
LAMBDA_1        = TARGET_MEAN / TARGET_VAR
LAMBDA_2        = -1./(2 * TARGET_VAR)
#GAMMA           = 1./(10*DELTA_T)         # leaking rate
EPS_W           = GAMMA/10     # learning rate
EPS_A           = GAMMA/10     # gain change rate
EPS_B           = GAMMA/10     # threshold change rate
w_mean          = 1
x_c             = 0.5

def activation(x, a, b):
    ''' sigmoidal '''
    return 1/(1 + np.exp(a*(b - x)))

def ErdosRenyi_network():
    '''create a random excitatory network, and then complete the graph with inhibitory links'''
    exc_adj = np.triu(np.random.rand(NEURONS, NEURONS) < PROB_CONN, k = 1) # upper triangular matrix above diagonal
    #exc_adj += exc_adj.T # symmetrize 
    inh_adj = np.triu(1 - exc_adj, k = 1)
    
    exc_weights = exc_adj * np.triu( np.random.normal(w_mean, 0.1*w_mean, (NEURONS, NEURONS)) )
    inh_weights = inh_adj * np.triu( np.random.normal(w_mean, 0.1*w_mean, (NEURONS, NEURONS)) )
    weights = (exc_adj - inh_adj) * np.random.normal(w_mean, 0.1*w_mean, (NEURONS, NEURONS))

    #weights = np.random.normal(0, 0.5, (NEURONS, NEURONS))
    weights += weights.T
    np.fill_diagonal(weights, 0)
    G = nx.Graph(weights, weight = weights)
    #pdb.set_trace()
    G_exc = nx.Graph(exc_adj+exc_adj.T)
    return weights, G, G_exc

def geometric_network(n_clique = 3, clique_size = 3, sparse = False):
    neurons = n_clique * clique_size
    single_clique = np.ones((clique_size, clique_size))
    exc_adj = LA.block_diag(*([single_clique]*n_clique))
    for i in range(n_clique):
        a, b = i*clique_size, i*clique_size-1
        exc_adj[a, b] = 1
        exc_adj[b, a] = 1
    exc_link_prob = exc_adj.sum()/(neurons*(neurons-1))
    if sparse:
        inh_adj = (1 - exc_adj) * (np.random.rand(neurons, neurons) < exc_link_prob) # sparse inhibitory connections with P_inh = P_exc
    else:
        inh_adj = (1 - exc_adj) * exc_link_prob/(1 - exc_link_prob) # all-to-all inhibitory connections with balanced weights
    inh_adj = np.triu(inh_adj, k = 1)
    inh_adj += inh_adj.T
    weights = w_mean * (exc_adj - inh_adj)/np.sqrt(clique_size + 2)
    np.fill_diagonal(weights, 0.)
    G = nx.Graph(weights, weight = weights)

    return neurons, weights, G

def rotating_clique_network(n_clique = 3, clique_size = 3, sparse = False):
    neurons = n_clique * clique_size
    exc_links = neurons * (clique_size +1) // 2
    tot_links = neurons*(neurons-1)//2
    A = np.zeros((neurons, neurons))
    for clique in range(n_clique):
        for i in range(clique_size):
            first = clique + i * n_clique
            for j in range(i+1, clique_size):
                second = clique + j * n_clique
                #print (first, second)
                A[first, second] = 1
    np.fill_diagonal(A[:, 1:], 1) 
    A[0, -1] = 1
    #G = nx.Graph(A)
    #nx.draw_circular(G, with_labels = True) 
    exc_link_prob = exc_links / tot_links
    #A += A.T
    if sparse:
        #inh_adj = (1 - A) * ( np.random.rand(neurons, neurons) < exc_link_prob ) # sparse inhibitory connections with P_inh = P_exc
        inh_adj = (1 - A) * ( np.random.rand(neurons, neurons) < exc_links/(tot_links - exc_links) ) # sparse inhibitory connections with <#inh> = #exc 
    else:
        inh_adj = (1 - A) * exc_link_prob/(1 - exc_link_prob) # all-to-all inhibitory connections with balanced weights
    inh_adj = np.triu(inh_adj, k = 1)
    weights = w_mean * (A - inh_adj)/np.sqrt(clique_size + 2)
    weights += weights.T
    np.fill_diagonal(weights, 0.)
    #pdb.set_trace()
    G = nx.Graph(weights, weight = weights)

    return neurons, weights, G

def activation(x, a, b):
    ''' sigmoidal '''
    return 1/(1 + np.exp(a*(b - x)))

def next_step():
    ''' Next step with Euler integration '''
    activity = activation(x, gain, threshold)
    recurrent_input = np.dot(weights, activity) # input vector
    noise = 0*np.random.uniform(-5, 5, size = NEURONS)
    x_inp = noise + recurrent_input
    dx = GAMMA*(x_inp - x)
    next_x = x + dx * DELTA_T

    aux = 1 - 2*activity + (LAMBDA_1 +2*LAMBDA_2*activity)*(1 - activity)*activity
    db = -EPS_B*gain*aux
    next_b = threshold + db * DELTA_T

    da = EPS_A*(1/gain + (x - threshold)*aux)*0
    next_a = np.maximum(gain + da * DELTA_T, 0.001)

    return next_x, next_b, next_a, activity, x_inp

def other_next_step():
    ''' Next step with Euler integration '''
    activity = activation(x, gain, threshold) 
    recurrent_input = np.dot(weights, activity) # input vector
    noise = 0*np.random.uniform(-5, 5, size = NEURONS)
    x_inp = noise + recurrent_input
    dx = GAMMA*(x_inp - x)
    next_x = x + dx * DELTA_T
    db = EPS_B * (activity - TARGET_MEAN)
    next_b = threshold + db * DELTA_T
    da = EPS_A * (1/gain - (activity - TARGET_MEAN)**2)
    next_a = np.maximum(gain + da * DELTA_T, 0.001)
    #dw = 0.  
    #next_w = weights + dw*DELTA_T
    return next_x, next_b, next_a, activity, x_inp


n_clique = 4 # at larger than 3 (for s_c = 3)
clique_size = 10 # larger than 2, otherwise cliques don't make sense
sparseness = True
if sparseness:
    if n_clique <= (2 * clique_size + 3)/clique_size:
        neurons = n_clique * clique_size
        exc_links = neurons * (clique_size +1) // 2
        tot_links = neurons*(neurons-1)//2
        p_inh = exc_links/(tot_links - exc_links)
        print('All-to-all even if sparse! P_inh = {}'.format(p_inh))

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
rotating_cycler = cycler('linestyle', ['-', '--', ':', '-.']) * cycler(color = colors)
rotating_cycler = rotating_cycler[:n_clique]

#NEURONS, weights, G = geometric_network(n_clique, clique_size, sparse = False)
NEURONS = 10
PROB_CONN = 0.3
#weights, G, G_exc = ErdosRenyi_network()

NEURONS, weights, G = rotating_clique_network(n_clique, clique_size, sparse = sparseness)
print(len(weights[weights>0])//2, len(weights[weights<0])//2)
first_clique = np.ones(clique_size)
#x = np.concatenate((first_clique, np.random.rand(NEURONS - clique_size)))
x = np.random.rand(NEURONS)
gain = np.random.normal(3, 0.5, NEURONS)
threshold = np.random.normal(0.5, 0.1, NEURONS)

if NEURONS < 20:
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
    
Y = np.zeros((NEURONS, SIM_STEPS))
inputs = np.zeros((NEURONS, SIM_STEPS))
B = np.zeros((NEURONS, SIM_STEPS))
A = np.zeros((NEURONS, SIM_STEPS))
X = np.zeros((NEURONS, SIM_STEPS))

for t in tqdm(range(SIM_STEPS)):
    x, threshold, gain, activity, x_inp = other_next_step()
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
fig_title = 'Activity of {} cliques with {} nodes, sparse = {}'.format(n_clique, clique_size, sparseness)
if NEURONS < 20:
    fig, axs = plt.subplots(NEURONS, 1, sharex=True)
    plt.suptitle(fig_title)
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
    axs[NEURONS-1].set_xlabel('time (s) + {}'.format((SIM_STEPS-final_times)/int(1000/DELTA_T)))


input_pl = inputs[:, -final_times:]
X_plot = X[:, -final_times:]
'''
fig_inp, axs_inp = plt.subplots(NEURONS, 1, sharex=True)
plt.suptitle('Input of {} cliques with {} nodes'.format(n_clique, clique_size))
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
fig, ax = plt.subplots()    
ax.set(ylabel = 'Activity y', xlabel = 'time (s) + {}'.format((SIM_STEPS-final_times)/int(1000/DELTA_T)), title = fig_title) #'excitatory fraction: {}, w_mean: {}'.format(EXC_FRAC, w_mean))
ax.set_prop_cycle(rotating_cycler)
ax.set_xlim(time_plot[-10000], time_plot[-1])
for i in range(neurons_plot):
    line, = ax.plot(time_plot, Y_plot[i].T, label = i%n_clique)
    #time.sleep(2)
    #ax.plot([time_plot[0], time_plot[-1]], [mean, mean], '--')
    #std_line, = ax.plot([time_plot[0], time_plot[-1]], [mean + std, mean + std], '--')
    #ax.plot([time_plot[0], time_plot[-1]], [mean - std, mean - std], '--', color = std_line.get_color())

save_figures = True
if save_figures:
    savefig_options = {'papertype' : 'a5', 'dpi' : 180}
    fig.savefig('{}x{}_s{}'.format(n_clique, clique_size, sparseness), **savefig_options)
    #plt.close(fig)
    #Y = 0
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