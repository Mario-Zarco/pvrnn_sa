import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle
from agent_acoustic import Agent
import numpy as np

np.random.seed(4)

# Environment
ENV_XB = 5.0
ENV_YB = 3.0

# Trials
T_TRIAL = 20.0
N_TRIALS = 1

# Time
dt = 0.1
time = np.arange(0, T_TRIAL, dt)

# Angular speed control
TAU_PHI = 0.5
SIGMA_PHI = 0.25

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
STIMULI_POSITIONS = np.random.uniform(-5, 5, size=(4, 2))
sort_indices = np.argsort(STIMULI_POSITIONS[:, 0])
STIMULI_POSITIONS = STIMULI_POSITIONS[sort_indices]
print(STIMULI_POSITIONS)
STIMULI_STRENGTH = 1.0

agent = Agent(body_radius=BODY_RADIUS, sensors_divergence_angle=SENSOR_DIVERGENCE_ANGLE,
              p_init=P_INIT, h_init=H_INIT, dt=dt)

agent.set_stimuli(STIMULI_POSITIONS, STIMULI_STRENGTH)

position_t = []
heading_t = []
phi_t = []
v_t = []
sensor_t = []

fig = plt.figure(figsize=(12, 6))
gs = GridSpec(3, 1)

ax = fig.add_subplot(gs[0, 0])
ax.grid(True)
ax.set_xlim(-10, 10)
ax.set_ylim(-5, 5)

for stimulus in STIMULI_POSITIONS:
    ax.plot(stimulus[0], stimulus[1], 'ro')

for _ in range(N_TRIALS):
    
    t = 0.0

    while agent.position[0] < ENV_XB:
        agent.step_proprioception(ENV_XB, ENV_YB, TAU_PHI, SIGMA_PHI, TAU_V, SIGMA_V)
        agent.step_exteroception(t)
        t += dt

        # print(agent.position, agent.heading)
        position_t.append(agent.position.copy())
        heading_t.append(agent.heading)
        sensor_t.append(agent.signal_strenghts)

        agent_body = Circle((agent.position[0], agent.position[1]), BODY_RADIUS, fill=False)
        ax.add_patch(agent_body)
        for sensor_position in agent._get_abs_sensors_position():
            ax.plot(sensor_position[0], sensor_position[1], 'ko')
    
    print("Final time", t)

ax = fig.add_subplot(gs[1, 0])
ax.grid(True)
ax.plot(sensor_t, '.')

plt.show()