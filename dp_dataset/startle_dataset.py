"""
- Set random stimulus properly
- Save all data
- Plot final data
"""

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle
import numpy as np
from agent import Agent

def wrap_angle(a):
    return (a + np.pi) % (2*np.pi) - np.pi

np.random.seed(42)

# Environment
ENV_XB = 5.0
ENV_YB = 3.0

# Trials
T_TRIAL = 20.0
N_TRIALS = 5

# Time
dt = 0.5
time = np.arange(0, T_TRIAL, dt)

# Angular speed control
# TAU_PHI = 0.5
# SIGMA_PHI = 0.25
TAU_PHI = 1.0
SIGMA_PHI = 0.5

# Linear speed control
TAU_V = 1.0
SIGMA_V = 0.5

# Agent
BODY_RADIUS = 0.25
SENSOR_DIVERGENCE_ANGLE = 30 * np.pi / 180
P_INIT = np.array([-ENV_XB + np.random.uniform(-0.5, 0.5),
                np.random.uniform(-1.0, 1.0)])
H_INIT = 0.0

# Stimulus
STIMULI_STRENGTH = 1.0

proprio_signal = []
extero_signal = []
intero_signal = []

# Initialize Agent
agent = Agent(body_radius=BODY_RADIUS, sensors_divergence_angle=SENSOR_DIVERGENCE_ANGLE,
            p_init=P_INIT, h_init=H_INIT, dt=dt, agent_type="DP")


for trial_n in range(N_TRIALS):

    stimuli_y_ub = np.random.normal(2, 0.27, 2)
    stimuli_y_lb = np.random.normal(-2, 0.25, 2)
    stimuli_x = np.random.uniform(-5, 5, 4)
    STIMULI_POSITIONS = np.column_stack((stimuli_x, np.concatenate((stimuli_y_lb, stimuli_y_ub))))

    agent.set_stimuli(STIMULI_POSITIONS, STIMULI_STRENGTH)
    
    t = 0.0

    while agent.position[0] < ENV_XB:
        agent.step_proprioception(ENV_XB, ENV_YB, TAU_PHI, SIGMA_PHI, TAU_V, SIGMA_V)
        agent.step_exteroception(t)
        agent.step_interoception()
        t += dt

        proprio_signal.append(np.array([agent.v, agent.phi]))
        extero_signal.append(agent.signal_strengths)
        intero_signal.append(agent.heart_activity)
    
    t_return = T_TRIAL - t
    steps_return = int(t_return / dt)

    p0 = agent.position.copy()
    next_start = np.array([-ENV_XB + np.random.uniform(-0.5, 0.5),
                            np.random.uniform(-1.0, 1.0)])
    
    delta = next_start - p0
    distance = np.linalg.norm(delta)
    direction = delta / (distance + 1e-8)
    theta_motion = np.arctan2(direction[1], direction[0])

    phi_start = agent.phi  # last forward value

    for k in range(steps_return):

        tau = (k + 1) / steps_return

        # Smooth velocity profile (bounded [0,1])
        v = 4 * tau * (1 - tau)

        # Smoothly drive phi to zero
        phi = (1 - tau) * phi_start

        # Bound to [-1,1]
        phi = np.clip(phi, -1.0, 1.0)

        agent.step_return(v, theta_motion)

        proprio_signal.append(np.array([v, phi]))
        extero_signal.append(agent.signal_strengths)
        intero_signal.append(agent.heart_activity)

    print("Trial:", trial_n, "Final time", t, "Returning time", t_return, "Total time", t + t_return)

proprio_signal = np.array(proprio_signal)
extero_signal = np.array(extero_signal)
intero_signal = np.array(intero_signal)

fig = plt.figure(figsize=(16, 6))
gs = GridSpec(3, 1)

ax = fig.add_subplot(gs[0, 0])
ax.grid(True)
ax.set_ylim(-1.1, 1.1)
ax.plot(proprio_signal[:, 0], '.-',label='v')
ax.plot(proprio_signal[:, 1], '.-',label='phi')
ax.legend()

ax = fig.add_subplot(gs[1, 0])
ax.grid(True)
ax.set_ylim(-0.1, 1.1)
ax.plot(extero_signal[:, 0], '.-',label='left sensor')
ax.plot(extero_signal[:, 1], '.-',label='right sensor')
ax.legend()

ax = fig.add_subplot(gs[2, 0])
ax.grid(True)
ax.set_ylim(-0.1, 1.1)
ax.plot(intero_signal, '.-',label='heart activity')
ax.legend()

plt.show()