"""
Coherent probabilistic locomotion loop
- Symmetric turn
- Positive varying velocity
- Bounded environment
- Smooth angular velocity
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from agent import Agent

np.random.seed(42)

# ------------------------
# Environment
# ------------------------

ENV_XB = 5.0
ENV_YB = 3.0

# ------------------------
# Trials
# ------------------------

T_TRIAL = 20.0
N_TRIALS = 5

# ------------------------
# Time
# ------------------------

dt = 0.1
time = np.arange(0, T_TRIAL, dt)

# ------------------------
# Agent parameters
# ------------------------

BODY_RADIUS = 0.25
SENSOR_DIVERGENCE_ANGLE = 30 * np.pi / 180

# Tuned for closed bounded loops
V_MEAN = 0.5 # 0.45
TAU_V = 1.5
SIGMA_V = 0.15

TAU_PHI = 1.0
SIGMA_PHI = 0.15
K_TURN = 1.0
TURN_STD = 0.1
TURN_WIDTH = 1.2

STIMULI_STRENGTH = 1.0

# ------------------------
# Helper OU update
# ------------------------

def ou_update(z, mu, tau, sigma, dt):
    return z + (mu - z)/tau * dt + sigma * np.sqrt(dt) * np.random.randn()


# ------------------------
# Storage
# ------------------------

proprio_signal = []
extero_signal = []
intero_signal = []
position_signal = []

# ------------------------
# Initialize Agent
# ------------------------

P_INIT = np.array([
    -ENV_XB + np.random.uniform(-0.5, 0.5),
    np.random.uniform(-1.0, 1.0)
])

agent = Agent(
    body_radius=BODY_RADIUS,
    sensors_divergence_angle=SENSOR_DIVERGENCE_ANGLE,
    p_init=P_INIT,
    h_init=0.0,
    dt=dt,
    agent_type=""
)

# ------------------------
# Trials
# ------------------------

for trial_n in range(N_TRIALS):

    # Random stimuli
    stimuli_y_ub = np.random.normal(2, 0.27, 2)
    stimuli_y_lb = np.random.normal(-2, 0.25, 2)
    stimuli_x = np.random.uniform(-5, 5, 4)

    STIMULI_POSITIONS = np.column_stack(
        (stimuli_x, np.concatenate((stimuli_y_lb, stimuli_y_ub)))
    )

    agent.set_stimuli(STIMULI_POSITIONS, STIMULI_STRENGTH)

    # Reset trial internal states
    agent.position = np.array([
        -ENV_XB + np.random.uniform(-0.5, 0.5),
        np.random.uniform(-1.0, 1.0)
    ])
    agent.heading = 0.0

    z_v = 0.0
    z_phi = 0.0

    # Symmetric probabilistic turn
    t_turn = T_TRIAL/2 + np.random.normal(0, TURN_STD)

    t_internal = 0.0

    while t_internal < T_TRIAL:

        # ------------------------
        # Smooth symmetric turning goal
        # ------------------------

        s = np.tanh((t_internal - t_turn) / TURN_WIDTH)
        theta_goal = 0.5 * np.pi * (1 + s)

        # Angular error (wrapped)
        angle_error = (theta_goal - agent.heading + np.pi) % (2*np.pi) - np.pi

        # OU angular velocity
        z_phi = ou_update(
            z_phi,
            K_TURN * angle_error,
            TAU_PHI,
            SIGMA_PHI,
            dt
        )

        phi = np.clip(z_phi, -1.0, 1.0)

        # ------------------------
        # Positive stochastic velocity
        # ------------------------

        z_v = ou_update(
            z_v,
            V_MEAN,
            TAU_V,
            SIGMA_V,
            dt
        )

        v = np.clip(z_v, 0.0, 1.0)

        # ------------------------
        # Integrate heading
        # ------------------------

        agent.heading += phi * dt

        # ------------------------
        # Update position
        # ------------------------

        agent.position += 2 * v * np.array([
            np.cos(agent.heading),
            np.sin(agent.heading)
        ]) * dt

        # ------------------------
        # Soft boundary constraint
        # ------------------------

        agent.position[0] = np.clip(agent.position[0], -ENV_XB, ENV_XB)
        agent.position[1] = np.clip(agent.position[1], -ENV_YB, ENV_YB)

        agent.v = v
        agent.phi = phi

        agent._update_sensors_positions()
        agent.step_exteroception(t_internal)
        agent.step_interoception()

        # ------------------------
        # Store signals
        # ------------------------

        proprio_signal.append([v, phi])
        extero_signal.append(agent.signal_strengths)
        intero_signal.append(agent.heart_activity)
        position_signal.append(agent.position.copy())

        t_internal += dt

    print("Trial:", trial_n, "Final position:", agent.position)


# ------------------------
# Convert to arrays
# ------------------------

proprio_signal = np.array(proprio_signal)
extero_signal = np.array(extero_signal)
intero_signal = np.array(intero_signal)
position_signal = np.array(position_signal)

# ------------------------
# Plotting
# ------------------------

fig = plt.figure(figsize=(16, 8))
gs = GridSpec(4, 1)

# Velocity & angular velocity
ax1 = fig.add_subplot(gs[0, 0])
ax1.grid(True)
ax1.set_ylim(-1.1, 1.1)
ax1.plot(proprio_signal[:, 0], label='v')
ax1.plot(proprio_signal[:, 1], label='phi')
ax1.set_title("Proprioception")
ax1.legend()

# Exteroception
ax2 = fig.add_subplot(gs[1, 0])
ax2.grid(True)
ax2.set_ylim(-0.1, 1.1)
ax2.plot(extero_signal[:, 0], label='left sensor')
ax2.plot(extero_signal[:, 1], label='right sensor')
ax2.set_title("Exteroception")
ax2.legend()

# Interoception
ax3 = fig.add_subplot(gs[2, 0])
ax3.grid(True)
ax3.set_ylim(-0.1, 1.1)
ax3.plot(intero_signal, label='heart activity')
ax3.set_title("Interoception")
ax3.legend()

# Trajectory
ax4 = fig.add_subplot(gs[3, 0])
ax4.grid(True)
ax4.set_xlim(-ENV_XB, ENV_XB)
ax4.set_ylim(-ENV_YB, ENV_YB)
ax4.set_title("Trajectory")

# positions = np.array([agent.position for _ in range(len(proprio_signal))])
ax4.plot(position_signal[:, 0], position_signal[:, 1])
# ax4.axvline(-ENV_XB, linestyle='--')
# ax4.axvline(ENV_XB, linestyle='--')

plt.tight_layout()
plt.show()
