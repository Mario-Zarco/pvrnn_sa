"""

"""
import numpy as np
from utils import ou_step


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
        
        # INTEROCEPTION
        if agent_type == "DP":
            self.alpha = 0.99
            self.beta = 0.01
        else: # Healthy
            self.alpha = 0.0
            self.beta = 1.0
        self.ans = 0.0
        self.heart_activity = 0.0 
        self.threat_p = None

        self._update_sensors_positions()

    def set_stimuli(self, stimuli_positions, stimuli_strength, stimulus_type="Random"):
        """
        Set world once
        If stimulus type is Random, on stimulus is active at a time with a random duration 
            and they are randomly selected to be perceived by the agent
        If stimulus type is not Random (Closest distance), all stimulus are active,
            but only the closest to the agent can be perceived
        """
        self.stimuli_positions = stimuli_positions
        self.stimuli_strength = stimuli_strength
        self.stimulus_type = stimulus_type

        if self.stimulus_type == "Random":
            n_stimuli = self.stimuli_positions.shape[0]
            # Random Indexes
            self.rand_stimuli_idxs = np.arange(0, n_stimuli, 1)
            # np.random.shuffle(self.rand_stimuli_idxs)
            # Random Times
            self.rand_stimuli_times = []
            count_stimulus = 0
            while count_stimulus < n_stimuli:
                t = np.random.normal(2.25, 0.25)
                if t > 0.0:
                    self.rand_stimuli_times.append(t)
                    count_stimulus += 1
            # Cumsum Times
            self.cum_stimuli_time = np.cumsum(self.rand_stimuli_times)
            # Global counter
            self.global_count_stimulus = 0
            # Total time
            self.total_stimuli_time = np.sum(self.rand_stimuli_times)
    
    # ------------------------
    # ---- PROPRIOCEPTION ----
    # ------------------------

    def step_proprioception(self, env_xb, env_yb, tau_phi, sigma_phi, tau_v, sigma_v):
        # OU step
        self.phi, self.v = self._get_ou_speeds(env_xb, env_yb, tau_phi, sigma_phi, tau_v, sigma_v)
        # Calculate heading and position
        self.heading = self.phi
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
        self.z_phi = ou_step(self.z_phi, tau_phi, sigma_phi, self.dt)
        self.z_v = ou_step(self.z_v, tau_v, sigma_v, self.dt)
        # Return bouned speeds
        return bounded_phi(env_xb, env_yb, self.z_phi), bounded_v(self.z_v)

    # -----------------------
    # ---- EXTEROCEPTION ----
    # -----------------------
    def step_exteroception(self, t):
        self.signal_strenghts = self._signal_strengths(t)
        # Collapse both signal stranghts
        self.sensor_signal = np.sum(self.signal_strenghts)

    def _signal_strengths(self, t):
        # TODO: Test this signal
        """
        Current t used when stimulys type in Random
        """
        # Get stimulus and distance to stimulus
        if self.stimulus_type == "Random":
            stimulus_position, stimulus_distance = self._get_random_stimulus(t)
        else: # Closest stimulus is the defaults option
            stimulus_position, stimulus_distance = self._get_closest_stimulus()

        signal_strengths = np.array([0.0, 0.0])

        # In the case of Random stimuli
        if stimulus_position is not None and stimulus_distance is not None:

            pow_D_centers = np.power(stimulus_distance, 2)

            for i, sensor_position in enumerate(self._get_abs_sensors_position()):
                distance_sensor_stimulus = np.linalg.norm(sensor_position - stimulus_position)
                # Assuming overlaping should not happen
                distance_sensor_stimulus = max(distance_sensor_stimulus, self.body_radius)
                N = distance_sensor_stimulus / self.body_radius
                Is = self.stimuli_strength / np.power(N, 2.0)

                pow_Radius = np.power(self.body_radius, 2)
                pow_Dsen = np.power(distance_sensor_stimulus, 2)
                A = (pow_D_centers - pow_Radius) / pow_Dsen
                Dsh = 0 if A >= 1 else distance_sensor_stimulus * (1 - A)

                attenuation_slope = 0.9 / (2 * self.body_radius)
                AttenuationFactor = 1 - attenuation_slope * Dsh
                TotalSignal = Is * AttenuationFactor
                signal_strengths[i] = TotalSignal 

        return signal_strengths

    def _get_closest_stimulus(self):
        distances = np.zeros(self.stimuli_positions.shape[0])
        for i, threat_position in enumerate(self.stimuli_positions):
            d_to_threat = np.linalg.norm(self.p - threat_position)
            distances[i] = d_to_threat
        idx = np.argmin(distances)
        return self.stimuli_positions[idx], distances[idx]
    
    def _get_random_stimulus(self, t):
        # TODO: Double check when testing
        # if self.cum_time[self.global_count_stimulus] < t:
        # Get stimulus
        stimulus_position = self.stimuli_positions[self.rand_stimuli_idxs[self.global_count_stimulus]]
        stimulus_distance = np.linalg.norm(stimulus_position - self.position)
        print(stimulus_position, self.global_count_stimulus, self.cum_stimuli_time[self.global_count_stimulus], t)
        # Check if change stimulus
        if self.cum_stimuli_time[self.global_count_stimulus] < t + self.dt:
            # Next stimulus only if there are more
            if self.global_count_stimulus < self.stimuli_positions.shape[0] - 1:
                self.global_count_stimulus += 1
            else: # If no more stimuli, return None
                return None, None
        return stimulus_position, stimulus_distance

    # -----------------------
    # ---- INTEROCEPTION ----
    # -----------------------

    def step_interoception(self):
        self._hidden_autonomic_nervous_system()
        self.heart_activity = np.tanh(self.ans)
    
    def _hidden_autonomic_nervous_system(self, stimulus_intensity):
        self.ans = self.alpha * self.ans + self.beta * np.tanh(stimulus_intensity/0.5)


if __name__ == "__main__":

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