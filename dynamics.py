#!/usr/bin/env python3
import numpy as np                  # for math

GAMMA           = 1./20         # leaking rate
EPS_A           = GAMMA/10     # gain change rate
EPS_B           = GAMMA/10     # threshold change rate
EPS_W           = GAMMA/100     # learning rate
TARGET_MEAN     = 0.3
TARGET_VAR      = 0.1**2
LAMBDA_1        = TARGET_MEAN / TARGET_VAR
LAMBDA_2        = -1./(2 * TARGET_VAR)


def activation(x, a, b):
    ''' sigmoidal '''
    return 1/(1 + np.exp(a*(b - x)))

def polyhomeostatic(x, gain, threshold, weights):
    ''' Next step with Euler integration '''
    activity = activation(x, gain, threshold)
    recurrent_input = np.dot(weights, activity) # input vector
    noise = 0*np.random.uniform(-5, 5, size = NEURONS)
    x_inp = noise + recurrent_input
    dx = GAMMA*(x_inp - x)
    #next_x = x + dx * DELTA_T

    aux = 1 - 2*activity + (LAMBDA_1 +2*LAMBDA_2*activity)*(1 - activity)*activity
    db = -EPS_B*gain*aux
    #next_b = threshold + db * DELTA_T

    da = EPS_A*(1/gain + (x - threshold)*aux)
    #next_a = np.maximum(gain + da * DELTA_T, 0.001)

    return dx, db, da, activity, x_inp

def target(x, gain, threshold, weights, gain_rule):
    ''' Next step with Euler integration '''
    activity = activation(x, gain, threshold) 
    recurrent_input = np.dot(weights, activity) # input vector
    #noise = 0*np.random.uniform(-5, 5, size = NEURONS)
    x_inp = recurrent_input# + noise
    dx = GAMMA*(x_inp - x)
    #next_x = x + dx * DELTA_T
    db = EPS_B * (activity - TARGET_MEAN)
    #next_b = threshold + db * DELTA_T
    if gain_rule == 1:
        da = EPS_A * (TARGET_VAR - (activity - TARGET_MEAN)**2)/gain
    elif gain_rule == 2:
        da = EPS_A * (1/gain - (activity - TARGET_MEAN)**2)
    else:
        da = np.zeros(NEURONS)
    #next_a = np.maximum(gain + da * DELTA_T, 0.001)
    #dw = 0.  
    #next_w = weights + dw*DELTA_T
    #return next_x, next_b, next_a, activity, x_inp
    return dx, db, da, activity, x_inp