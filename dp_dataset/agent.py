"""

"""
import numpy as np
from utils import ou_step
from utils import ou_update


def bounded_phi(xb, yb, z):
    return np.arcsin(yb / xb) * np.tanh(z)


def bounded_v(z):
    return 0.5 * (np.tanh(z) + 1.0)


class Agent:
    def __init__(self, body_radius: float, sensors_divergence_angle: float, 
                 p_init: np.array, h_init: float, dt: float, agent_type: str=""):
        self.dt = dt

        self.body_radius = body_radius
        self.sensors_divergence_angle = sensors_divergence_angle
        self.sensors_positions = None

        # PROPRIOCEPTION
        self.position = p_init.copy()
        self.heading = h_init
        self.phi = 0
        self.v = 0
        self.z_phi = 0 # np.random.randn()
        self.z_v = 0 # np.random.randn()

        # EXTEROCEPTION
        self.stimuli_positions = None
        self.stimuli_strength = 0
        self.stimulus_type = ""
        # Signals from both sensors are collapsed to a single signal
        self.sensor_signal = None 

        self.sound_range = 5.0          # lambda for exp distance decay
        self.occlusion_center = 0.0     # cos(theta) threshold
        self.occlusion_slope = 0.15     # smoothness of occlusion
        self.max_signal = 1.0           # stimulus peak strength
        
        # INTEROCEPTION
        if agent_type == "DP":
            self.alpha = 0.98
            self.beta = 0.02
        else: # Healthy
            self.alpha = 0.0
            self.beta = 1.0
        self.ans = 0.0
        self.heart_activity = 0.0 
        self.threat_p = None

        self._update_sensors_positions()

    def set_stimuli(self, stimuli_positions, stimuli_strength):
        self.stimuli_positions = stimuli_positions
        self.stimuli_strength = stimuli_strength

        n = stimuli_positions.shape[0]

        # Sample durations (positive only)
        durations = []
        while len(durations) < n:
            T = np.random.normal(2.25, 0.25)
            if T > 0:
                durations.append(T)

        self.stim_durations = np.array(durations)
        self.stim_start_times = np.concatenate([[0.0], np.cumsum(self.stim_durations[:-1])])
        self.stim_end_times = self.stim_start_times + self.stim_durations

        self.active_stimulus = 0

    
    # ------------------------
    # ---- PROPRIOCEPTION ----
    # ------------------------

    def step_proprioception(self, env_xb, env_yb, tau_phi, sigma_phi, tau_v, sigma_v):
        # OU step
        self.phi, self.v = self._get_ou_speeds(env_xb, env_yb, tau_phi, sigma_phi, tau_v, sigma_v)
        # Calculate heading and position
        self.heading = 1 * self.phi
        # self.heading += self.phi * self.dt
        self.position += 2 * self.v * np.array([np.cos(self.heading), np.sin(self.heading)]) * self.dt
        # Update sensors position
        self._update_sensors_positions()

    def _update_sensors_positions(self):
        sensors_angles = np.array([self.heading + self.sensors_divergence_angle, self.heading - self.sensors_divergence_angle])
        self.sensors_positions = np.array([self.body_radius * np.array([np.cos(angle), np.sin(angle)]) for angle in sensors_angles])
    
    def _get_abs_sensors_position(self):
        return [self.position + ep for ep in self.sensors_positions]
    
    def _get_ou_speeds(self, env_xb, env_yb, tau_phi, sigma_phi, tau_v, sigma_v):
        # Get out speeds
        # self.z_phi = ou_step(self.z_phi, tau_phi, sigma_phi, self.dt)
        # self.z_v = ou_step(self.z_v, tau_v, sigma_v, self.dt)
        self.z_phi = ou_update(self.z_phi, 0, tau_phi, sigma_phi, self.dt)
        self.z_v = ou_update(self.z_v, 0, tau_v, sigma_v, self.dt)
        # Return bouned speeds
        return bounded_phi(env_xb, env_yb, self.z_phi), bounded_v(self.z_v)
    
    def step_return(self, v, theta_motion):
        self.position += 2 * v * np.array([np.cos(theta_motion),
                                       np.sin(theta_motion)]) * self.dt
        # self.position += 2 * v * np.array([np.cos(phi), np.sin(phi)]) * self.dt
        self.sensor_signal = 0.0
        self.ans = 0.0

    # -----------------------
    # ---- EXTEROCEPTION ----
    # -----------------------

    def step_exteroception(self, t):
        self.signal_strengths = self._signal_strengths(t)

        # Soft fusion (smooth max)
        # k = 5.0
        # fused = np.log(np.sum(np.exp(k * self.signal_strengths))) / k

        # self.sensor_signal = np.clip(fused, 0.0, 1.0)
        self.sensor_signal = np.mean(self.signal_strengths)

    def _signal_strengths(self, t):
        signal = np.zeros(2)

        if self.active_stimulus >= len(self.stimuli_positions):
            return signal

        t_start = self.stim_start_times[self.active_stimulus]
        t_end = self.stim_end_times[self.active_stimulus]

        if t > t_end:
            self.active_stimulus += 1
            return signal

        # Temporal envelope
        T = self.stim_durations[self.active_stimulus]
        env = self._temporal_envelope(t, t_start, T)

        if env == 0.0:
            return signal

        stim_pos = self.stimuli_positions[self.active_stimulus]

        for i, sensor_pos in enumerate(self._get_abs_sensors_position()):
            vec = stim_pos - sensor_pos
            d = np.linalg.norm(vec)

            # if d < 1e-6:
            #     continue

            stim_dir = vec / d
            sensor_dir = np.array([
                np.cos(self.heading + (-1)**i * self.sensors_divergence_angle),
                np.sin(self.heading + (-1)**i * self.sensors_divergence_angle)
            ])

            gain_d = self._distance_gain(d)
            gain_o = self._occlusion_factor(sensor_dir, stim_dir)

            signal[i] = self.max_signal * env * gain_d * gain_o

        return signal # np.clip(signal, 0.0, 1.0)

    def _distance_gain(self, d):
        return np.exp(-d / self.sound_range)

    def _occlusion_factor(self, sensor_dir, stim_dir):
        cos_theta = np.dot(sensor_dir, stim_dir)
        x = (cos_theta - self.occlusion_center) / self.occlusion_slope
        return 1.0 / (1.0 + np.exp(-x))
    
    def _temporal_envelope(self, t, t_start, T):
        tau = (t - t_start) / T
        if tau < 0.0 or tau > 1.0:
            return 0.0
        return np.sin(np.pi * tau) ** 2

    # -----------------------
    # ---- INTEROCEPTION ----
    # -----------------------

    def step_interoception(self):
        self._hidden_autonomic_nervous_system(self.sensor_signal)
        self.heart_activity = np.tanh(self.ans)
    
    def _hidden_autonomic_nervous_system(self, stimulus_intensity):
        self.ans = self.alpha * self.ans + self.beta * np.tanh(stimulus_intensity/0.5)


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Circle
    import numpy as np

    # np.random.seed(42)

    # Environment
    ENV_XB = 5.0
    ENV_YB = 3.0

    # Trials
    T_TRIAL = 20.0
    N_TRIALS = 1

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
    # STIMULI_POSITIONS = np.random.uniform(-5, 5, size=(4, 2))
    stimuli_y_ub = np.random.normal(2, 0.25, 2)
    stimuli_y_lb = np.random.normal(-2, 0.25, 2)

    stimuli_y = np.concatenate((stimuli_y_lb, stimuli_y_ub))
    
    np.random.shuffle(stimuli_y)

    stimuli_x = np.linspace(-4, 4, 4)
    stimuli_x += np.random.normal(0, 0.25, 4)

    np.random.shuffle(stimuli_x)

    positions = np.column_stack((stimuli_x, stimuli_y))

    STIMULI_POSITIONS = positions

    # sort_indices = np.argsort(STIMULI_POSITIONS[:, 0])
    # STIMULI_POSITIONS = STIMULI_POSITIONS[sort_indices]
    # print(STIMULI_POSITIONS)
    STIMULI_STRENGTH = 1.0

    agent = Agent(body_radius=BODY_RADIUS, sensors_divergence_angle=SENSOR_DIVERGENCE_ANGLE,
                p_init=P_INIT, h_init=H_INIT, dt=dt, agent_type="")

    agent.set_stimuli(STIMULI_POSITIONS, STIMULI_STRENGTH)

    position_t = []
    heading_t = []
    phi_t = []
    v_t = []
    sensor_t = []
    intero_t = []

    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(4, 1)

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
            agent.step_interoception()
            t += dt

            # print(agent.position, agent.heading)
            position_t.append(agent.position.copy())
            heading_t.append(agent.heading)
            sensor_t.append(agent.signal_strengths)
            # sensor_t.append(agent.sensor_signal)
            intero_t.append(agent.heart_activity)
            phi_t.append(agent.phi)
            v_t.append(agent.v)

            agent_body = Circle((agent.position[0], agent.position[1]), BODY_RADIUS, fill=False)
            ax.add_patch(agent_body)
            for sensor_position in agent._get_abs_sensors_position():
                ax.plot(sensor_position[0], sensor_position[1], 'ko')
        
        print("Final time", t)

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

            phi_t.append(agent.phi)
            v_t.append(agent.v)
            sensor_t.append(agent.signal_strengths)
            intero_t.append(agent.heart_activity)

            agent_body = Circle((agent.position[0], agent.position[1]), BODY_RADIUS, fill=False)
            ax.add_patch(agent_body)
            for sensor_position in agent._get_abs_sensors_position():
                ax.plot(sensor_position[0], sensor_position[1], 'ko')

    ax = fig.add_subplot(gs[1, 0])
    ax.grid(True)
    ax.plot(sensor_t, '.-')

    ax = fig.add_subplot(gs[2, 0])
    ax.grid(True)
    ax.plot(intero_t, '.-')
    ax.set_ylim(-0.1, 1.1)

    ax = fig.add_subplot(gs[3, 0])
    ax.grid(True)
    ax.plot(phi_t, '.-', label='phi')
    ax.plot(v_t, '.-', label='v')
    ax.legend()

    plt.show()