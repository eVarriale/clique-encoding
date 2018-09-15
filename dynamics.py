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

    #inhib_input = np.dot(weights * (weights<0), activity)
    dV, nwc, c = sensory_plasticity(sensory_inp, recurrent_input,
                                    membrane_potential, sensory_signal, dx,
                                    sensory_weights, activity)
    return dx, db, da, activity, x_inp, sensory_inp, dV

U_max = 4.
T_u_inh = 30. #10.
T_φ_inh = 60. #20.

T_u_exc = 5 * T_u_inh
T_φ_exc = 5 * T_φ_inh
T_x = 20.

gain = 10# default : 10

def full_depletion(membrane_potential, full_vesicles_inh, vesic_release_inh, 
                   full_vesicles_exc, vesic_release_exc, excit_weights, 
                   inhib_weights, sensory_signal=None, sensory_weights=None):

    ''' Next step with Euler integration '''

    activity = activation(membrane_potential, gain)
    noise = 1#np.random.normal(1, 0.01, activity.shape)
    effective_activity_inh = activity * full_vesicles_inh * vesic_release_inh
    effective_activity_exc = activity * full_vesicles_exc * vesic_release_exc
    # This enforces Tsodyks-Markram rule I_j = sum_k z_jk phi_k u_k y_k
    #excit_input = np.dot(excit_weights, activity*noise)
    excit_input = np.dot(excit_weights, effective_activity_exc * noise)
    inhib_input = np.dot(inhib_weights, effective_activity_inh * noise)
    sensory_inp = np.dot(sensory_weights, sensory_signal)
    total_input = excit_input + inhib_input + sensory_inp

    dx = (total_input - membrane_potential) / T_x

    U_y = 1 + (U_max - 1) * activity
    d_u_inh = (U_y - vesic_release_inh) / T_u_inh
    d_u_exc = (U_y - vesic_release_exc) / T_u_exc

    ϕ_u_inh = 1 - vesic_release_inh * activity / U_max
    d_φ_inh = (ϕ_u_inh - full_vesicles_inh) / T_φ_inh
    ϕ_u_exc = 1 - vesic_release_exc * activity / U_max
    d_φ_exc = (ϕ_u_exc - full_vesicles_exc) / T_φ_exc

    #if (sensory_signal == 0.).all():
    #    dV = learning = decay = np.zeros(sensory_weights.shape)
    #    new_winning_clique = c = np.zeros(membrane_potential.shape)

    #else:
    dV, learning, decay = sensory_plasticity(sensory_inp, inhib_input+excit_input,
                                    membrane_potential, sensory_signal, dx,
                                    sensory_weights, activity)
    #pdb.set_trace()
    #return dx, d_u_inh, d_φ_inh, activity, total_input, sensory_inp, dV, learning, decay
    return dx, d_u_inh, d_φ_inh, d_u_exc, d_φ_exc, activity, total_input, sensory_inp, dV, learning, decay

T_l = 2000
T_f = 10 * 60 * 1000
T_v = 1

def sensory_plasticity(sensory_inp, recur_inp, membrane_potential, sensory_signal, 
                       dx, sensory_weights, activity):
    #
    V_ina = 0.3
    V_act = 0.8#8

    if (sensory_signal != 0.).any():
        V_target = V_ina + activity * (V_act - V_ina)
        c = 1 * np.tanh(10*(V_target - sensory_inp))
        #new_winning_clique = sensory_inp + inhib_input 
        # standard rule

        #new_winning_clique = sensory_inp + recur_inp

        new_winning_clique = 1# - inhib_input - excit_input

        #new_winning_clique = activation(sensory_inp + inhib_input, 10, 0.1)#, 0.5)
        
        #new_winning_clique = 1 - np.heaviside((sensory_inp+recur_inp)*recur_inp, .5)
        #new_winning_clique = 1 - activation((sensory_inp+recur_inp)*recur_inp, 100)
        
        #new_winning_clique = np.sign(sensory_inp + inhib_input)

        dy = gain * activity * (1 - activity) * dx
        #np.maximum(dy, 0 , dy)
        learning = np.outer(dy * new_winning_clique * c, sensory_signal)
        #pdb.set_trace()
    else:
        learning = np.zeros(sensory_weights.shape)
        new_winning_clique = c = np.zeros(dx.shape)
    
    #losing_cliques = 1 - new_winning_clique
    #decay =  - np.outer(losing_cliques, sensory_signal)
    decay = -1 * sensory_signal
    #w_target = 0.0#5

    dV = (learning / T_v + decay / T_f) * sensory_weights# - decay * w_target / T_f
    return dV, learning, decay

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
    return vector_input, V + H