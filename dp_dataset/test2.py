import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

np.random.seed(42)

# =========================
# Agent Class
# =========================

class Agent:

    def __init__(self, p_init, dt, env_xb, env_yb):

        self.position = np.array(p_init, dtype=float)
        self.heading = 0.0
        self.dt = dt

        self.env_xb = env_xb
        self.env_yb = env_yb

        # latent states
        self.z_v = 0.0
        self.z_phi = 0.0

        self.v = 0.0
        self.phi = 0.0

    # -------------------------
    # OU
    # -------------------------

    def ou_update(self, z, mu, tau, sigma):
        return z + (mu - z)/tau * self.dt + sigma*np.sqrt(self.dt)*np.random.randn()

    # -------------------------
    # Smooth bounds
    # -------------------------

    def bound_velocity(self, z):
        return 0.5 * (1 + np.tanh(z))  # (0,1)

    def bound_phi(self, z):
        return np.tanh(z)  # (-1,1)

    # -------------------------
    # Boundary steering
    # -------------------------

    def boundary_turn(self, gain=3.0, margin=0.8):

        x, y = self.position
        turn = 0.0

        # right wall
        if x > self.env_xb - margin:
            turn += gain * np.tanh(x - (self.env_xb - margin))

        # left wall
        if x < -self.env_xb + margin:
            turn -= gain * np.tanh((-self.env_xb + margin) - x)

        # top wall
        if y > self.env_yb - margin:
            turn += gain * np.tanh(y - (self.env_yb - margin))

        # bottom wall
        if y < -self.env_yb + margin:
            turn -= gain * np.tanh((-self.env_yb + margin) - y)

        return turn

    # -------------------------
    # Step
    # -------------------------

    def step(self, t, T_trial,
             v_mean=0.45,
             tau_v=1.5,
             sigma_v=0.15,
             tau_phi=1.0,
             sigma_phi=0.15,
             k_turn=3.0,
             turn_width=1.2,
             t_turn=None):

        if t_turn is None:
            t_turn = T_trial/2

        # Symmetric turning goal
        s = np.tanh((t - t_turn) / turn_width)
        theta_goal = 0.5 * np.pi * (1 + s)

        angle_error = (theta_goal - self.heading + np.pi) % (2*np.pi) - np.pi

        # Angular OU
        self.z_phi = self.ou_update(
            self.z_phi,
            k_turn * angle_error,
            tau_phi,
            sigma_phi
        )

        # Add boundary steering
        self.z_phi += self.boundary_turn()

        self.phi = self.bound_phi(self.z_phi)

        # Velocity OU
        self.z_v = self.ou_update(
            self.z_v,
            v_mean,
            tau_v,
            sigma_v
        )

        self.v = self.bound_velocity(self.z_v)

        # Integrate heading
        self.heading += self.phi * self.dt

        # Position update (pure kinematic)
        self.position += 2 * self.v * np.array([
            np.cos(self.heading),
            np.sin(self.heading)
        ]) * self.dt


# =========================
# Simulation
# =========================

ENV_XB = 5.0
ENV_YB = 3.0
T_TRIAL = 20.0
dt = 0.05
time = np.arange(0, T_TRIAL, dt)
N_TRIALS = 3

positions_all = []
proprio_all = []

for trial in range(N_TRIALS):

    p_init = [
        -ENV_XB + np.random.uniform(-0.5, 0.5),
        np.random.uniform(-1.0, 1.0)
    ]

    agent = Agent(p_init, dt, ENV_XB, ENV_YB)

    t_turn = T_TRIAL/2 + np.random.normal(0, 1.0)

    traj = []

    for t in time:
        agent.step(t, T_TRIAL, t_turn=t_turn)

        traj.append(agent.position.copy())
        proprio_all.append([agent.v, agent.phi])

    positions_all.append(np.array(traj))

proprio_all = np.array(proprio_all)

# =========================
# Plotting
# =========================

fig = plt.figure(figsize=(14, 10))
gs = GridSpec(3, 1)

# Signals
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(proprio_all[:, 0], label="v")
ax1.plot(proprio_all[:, 1], label="phi")
ax1.set_ylim(-1.1, 1.1)
ax1.grid(True)
ax1.legend()
ax1.set_title("Proprioception Signals")

# Trajectories
ax2 = fig.add_subplot(gs[1:, 0])
ax2.set_xlim(-ENV_XB, ENV_XB)
ax2.set_ylim(-ENV_YB, ENV_YB)
ax2.grid(True)
ax2.set_title("Agent Trajectories")

for traj in positions_all:
    ax2.plot(traj[:, 0], traj[:, 1])

plt.tight_layout()
plt.show()
