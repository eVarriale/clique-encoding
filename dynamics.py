#!/usr/bin/env python3
import numpy as np                  # for math
import random                       # for bars problem
import pdb

GAMMA           = 1./20         # leaking rate
EPS_A           = GAMMA/10     # gain change rate
EPS_B           = GAMMA/10     # threshold change rate
EPS_W           = GAMMA/100     # learning rate
TARGET_MEAN     = 0.3
TARGET_VAR      = 0.4**2
LAMBDA_1        = TARGET_MEAN / TARGET_VAR
LAMBDA_2        = -1./(2 * TARGET_VAR)

def activation(x, a=1., b=0.):
    ''' sigmoidal '''
    return 1/(1 + np.exp(a*(b - x)))

def polyhomeostatic(x, gain, threshold, weights):
    ''' Next step with Euler integration '''
    activity = activation(x, gain, threshold)
    recurrent_input = np.dot(weights, activity) # input vector
    x_inp = recurrent_input
    dx = GAMMA*(x_inp - x)
    #next_x = x + dx * DELTA_T

    aux = 1 - 2*activity + (LAMBDA_1 +2*LAMBDA_2*activity)*(1 - activity)*activity
    db = -EPS_B*gain*aux
    #next_b = threshold + db * DELTA_T

    da = EPS_A*(1/gain + (x - threshold)*aux)
    #next_a = np.maximum(gain + da * DELTA_T, 0.001)

    return dx, db, da, activity, x_inp

def target(membrane_potential, gain, threshold, weights, gain_rule, sensory_signal, sensory_weights):
    ''' Next step with Euler integration '''
    activity = activation(membrane_potential, gain, threshold)
    recurrent_input = np.dot(weights, activity)
    sensory_inp = np.dot(sensory_weights, sensory_signal)
    x_inp = recurrent_input + sensory_inp

    dx = GAMMA * (x_inp - membrane_potential)
    db = EPS_B * (activity - TARGET_MEAN)
    if gain_rule == 1:
        da = EPS_A * (TARGET_VAR - (activity - TARGET_MEAN)**2)/gain
    elif gain_rule == 2:
        da = EPS_A * (1/gain - (activity - TARGET_MEAN)**2)
    elif gain_rule == 3:
        da = 0.

    inhib_input = np.dot(weights * (weights<0), activity)
    dV = sensory_plasticity(sensory_inp, inhib_input, sensory_signal, dx, sensory_weights, activity)# - 0.0001 * sensory_weights

    return dx, db, da, activity, x_inp, sensory_inp, dV

U_max = 4.
T_u = 30. #10.

T_φ = 60. #20.
T_x = 20.
T_v = 100.
gain = 10

def full_depletion(membrane_potential, full_vesicles, vesic_release, excit_weights, inhib_weights, sensory_signal=None, sensory_weights=None):

    ''' Next step with Euler integration '''

    activity = activation(membrane_potential, gain)
    # effective_inhib_weights = inhib_weights * full_vesicles * vesic_release
    effective_activity = activity * full_vesicles * vesic_release
    # This enforces Tsodyks-Markram rule I_j = sum_k z_jk phi_k u_k y_k
    excit_input = np.dot(excit_weights, activity)
    #excit_input = np.dot(excit_weights, effective_activity)
    inhib_input = np.dot(inhib_weights, effective_activity)
    sensory_inp = np.dot(sensory_weights, sensory_signal)
    total_input = excit_input + inhib_input + sensory_inp

    dx = (total_input - membrane_potential) / T_x

    U_y = 1 + (U_max -1) * activity
    d_vesic_release = (U_y - vesic_release) / T_u

    ϕ_u = 1 - vesic_release * activity / U_max
    d_full_vesicles = (ϕ_u - full_vesicles) / T_φ

    #dV = 0.001 * np.outer( dx * activation(ext_inp - recurrent_input, 4, 0), ext_signal )# - 0.0001 * ext_weights
    if (sensory_signal == 0.).all():
        dV = np.zeros(sensory_weights.shape)
    else:
        dV = sensory_plasticity(sensory_inp, inhib_input, sensory_signal, dx, sensory_weights, activity)
    #pdb.set_trace()
    return dx, d_vesic_release, d_full_vesicles, activity, total_input, sensory_inp, dV

T_l = 1000
T_f = 60000

def sensory_plasticity(sensory_inp, inhib_input, sensory_signal, dx, sensory_weights, activity):
    #
    V_ina = 0.2
    V_act = 0.9
    V_target = V_ina + activity * (V_act - V_ina)
    c = np.tanh(10*(V_target - sensory_inp))
    new_winning_clique = sensory_inp + inhib_input
    #positive = 0.001 * np.outer(dx * activation(new_winning_clique, 10, 0) * c , sensory_signal)
    
    positive = np.outer(dx * new_winning_clique * c, sensory_signal) / T_l
    #negative = 0.0001 *  np.outer(dx * activation(-new_winning_clique, 10, 0), 1 - sensory_signal)
    #dV =  (positive - negative) * sensory_weights 
    losing_cliques = 1 - new_winning_clique
    negative = - np.outer(losing_cliques * c, sensory_signal) / T_l
    dV = (positive + 0*negative)* sensory_weights# - sensory_weights / T_f
    return dV

def bars_input(bar_size, prob):
    ''' Input from the bar_size x bar_size bars problem, 
    where each bar is present with probability prob
    2 * bar_size must be equal to n_clique! '''
    vert_bars = np.random.rand(bar_size) < prob
    hor_bars = np.random.rand(bar_size) < prob
    y = np.zeros((bar_size, bar_size))
    y[:, vert_bars] = 1
    y[hor_bars, :] = 1
    vector_input = y.ravel()

    V = vert_bars.sum()
    H = hor_bars.sum()

    assert y.sum() == (V + H) * bar_size - V * H, 'bars_input error'

    #pdb.set_trace()
    return vector_input