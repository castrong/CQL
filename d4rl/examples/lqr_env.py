import numpy as np
import gym
from gym import spaces
from typing import Optional, Tuple


class LQREnv(gym.Env):
    """
    Linear Quadratic Regulator environment.

    State: x ∈ R^n
    Control: u ∈ R^m
    Dynamics: ẋ = Ax + Bu
    Cost: c(x,u) = x^T Q x + u^T R u

    RL formulation:
    - Reward = -cost (to maximize reward = minimize cost)
    - Episode terminates after max_steps
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        Q: np.ndarray,
        Q_f: np.ndarray,
        R: np.ndarray,
        x_bounds: np.ndarray = None,
        u_bounds: np.ndarray = None,
        max_steps: int = 200,
        dt: float = 0.01,
        terminal_cost_weight: float = 1.0,
        render_mode: Optional[str] = None,
        max_episode_steps=10000,
        normalize_reward=False
    ):
        """
        Args:
            A: System matrix (n x n)
            B: Input matrix (n x m)
            Q: State cost matrix (n x n), positive semi-definite
            R: Control cost matrix (m x m), positive definite
            x_bounds: State bounds [x_min, x_max] for each dimension (n x 2)
            u_bounds: Control bounds [u_min, u_max] for each dimension (m x 2)
            max_steps: Maximum episode length
            dt: Time step for integration
            terminal_cost_weight: Weight for terminal cost x(T)^T Q x(T)
            render_mode: Rendering mode
        """
        super().__init__()

        self.A = A
        self.B = B
        self.Q = Q
        self.Q_f = Q_f
        self.R = R
        self.dt = dt
        self.max_steps = max_steps
        self.terminal_cost_weight = terminal_cost_weight
        self.render_mode = render_mode
        self._max_episode_steps = max_steps

        # State and control dimensions
        self.n_states = A.shape[0]
        self.n_controls = B.shape[1]

        # Set bounds
        if x_bounds is None:
            x_bounds = np.array([[-10.0, 10.0]] * self.n_states)
        if u_bounds is None:
            u_bounds = np.array([[-5.0, 5.0]] * self.n_controls)

        self.x_bounds = x_bounds
        self.u_bounds = u_bounds

        # Store original control bounds for scaling
        self.u_min = u_bounds[:, 0]
        self.u_max = u_bounds[:, 1]

        # Define action space as normalized [-1, 1] (SB3 recommendation)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_controls,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=x_bounds[:, 0],
            high=x_bounds[:, 1],
            shape=(self.n_states,),
            dtype=np.float32
        )

        # Internal state
        self.state = None
        self.steps = 0
        self.cumulative_cost = 0.0

        # calculate a reward shift and scale to normalize it
        # as a heuristic, we'll see the reward at the top right corner 
        # and bottom left corner of the state space, with maximum scaled actions
        # and then use the average of those two

        # Note: this heuristic won't account for the terminal cost 
        # being substantially larger, so that may sway things
        if normalize_reward:
            u_large = u_bounds[:, 1]
            reward_top = -self.dt * self._running_cost(x_bounds[:, 1], u_large) 
            reward_bottom = -self.dt * self._running_cost(x_bounds[:, 0], u_large) 
            avg_r = (reward_top + reward_bottom) / 2
            self.r_shift  = avg_r / 2
            self.r_scale = np.abs(avg_r) / 2
        else:
            self.r_shift = 0.0 
            self.r_scale = 1.0


    def normalize_reward(self, r):
        return (r - self.r_shift) / self.r_scale
    
    def normalize_cost(self, c):
        return (c + self.r_shift) / self.r_scale

    def _running_cost(self, x: np.ndarray, u: np.ndarray, normalize=True) -> float:
        """Compute running cost c(x,u) = x^T Q x + u^T R u"""
        return float(x.T @ self.Q @ x + u.T @ self.R @ u)

    def _terminal_cost(self, x: np.ndarray, normalize=True) -> float:
        """Compute terminal cost c_f(x) = x^T Q x"""
        return float(x.T @ self.Q_f @ x) * self.terminal_cost_weight

    def _dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Compute ẋ = Ax + Bu"""
        return self.A @ x + self.B @ u

    def set_state(self, x: np.ndarray):
        self.state = x

    def reset(
        self,
        #seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Reset the environment to initial state."""
        #super().reset()

        # Sample random initial state within bounds
        if options is not None and "initial_state" in options:
            self.state = np.array(options["initial_state"], dtype=np.float32)
        else:
            self.state = self.np_random.uniform(
                low=self.x_bounds[:, 0] * 0.8,  # Start closer to center
                high=self.x_bounds[:, 1] * 0.8,
                size=(self.n_states,)
            ).astype(np.float32)

        self.steps = 0
        self.cumulative_cost = 0.0

        info = {
            "cumulative_cost": self.cumulative_cost,
            "time": 0.0,
        }

        return self.state.copy() # , info

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]


    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one time step.

        Returns:
            observation: Next state
            reward: -cost (negative of running cost)
            terminated: Whether episode ended naturally
            truncated: Whether episode was cut off
            info: Additional information
        """
        # Scale action from [-1, 1] to actual control bounds
        # action_scaled = (action + 1) / 2 * (u_max - u_min) + u_min
        action = np.clip(action, -1.0, 1.0)  # Ensure within normalized bounds
        action_scaled = (action + 1.0) / 2.0 * (self.u_max - self.u_min) + self.u_min

        # Compute running cost with scaled action
        running_cost = self.normalize_cost(self.dt * self._running_cost(self.state, action_scaled))
        self.cumulative_cost += running_cost

        # Integrate dynamics (Euler integration) with scaled action
        x_dot = self._dynamics(self.state, action_scaled)
        self.state = self.state + x_dot * self.dt

        # Clip state to bounds
        #self.state = np.clip(self.state, self.x_bounds[:, 0], self.x_bounds[:, 1])

        self.steps += 1

        # Check termination conditions
        terminated = False
        # if any of the states have been clipped to their bounds, terminate 
        # for i in range(len(self.state)):
        #     if self.state[i] <= self.x_bounds[i, 0] or self.state[i] >= self.x_bounds[i, 1]:
        #         terminated = True 
        #         print("terminating at state: ", self.state)
        
        # if terminated == True:
        #     terminal_cost = self.normalize_cost(self._terminal_cost(self.state))
        #     self.cumulative_cost += terminal_cost
        #     # TODO: what to do for reward when terminated by hitting the boundary?
        #     # it should be more than the terminal cost
        #     reward = -(running_cost + terminal_cost)

        #else:
        truncated = False
        if self.steps >= self.max_steps:
            truncated = True
            # Add terminal cost
            terminal_cost = self.normalize_cost(self._terminal_cost(self.state))
            self.cumulative_cost += terminal_cost
            # Penalize terminal cost in this step's reward
            reward = -(running_cost + terminal_cost)
        else:
            # Reward is negative cost (we want to minimize cost)
            reward = -running_cost

        info = {
            "cumulative_cost": self.cumulative_cost,
            "running_cost": running_cost,
            "time": self.steps * self.dt,
            "state_norm": np.linalg.norm(self.state),
            "action_norm": np.linalg.norm(action_scaled),
        }

        # shift reward to be roughly 0 centered, at least for running cost
        # print("state: ", self.state)
        # print("a: ", action_scaled)
        # print("reward: ", reward)
        # if reward < 0.0:
            # print("!!!!!!!!!!!!!!!!")
        return self.state.copy(), float(reward), terminated or truncated, info

    def render(self):
        """Render the environment (optional)."""
        if self.render_mode == "human":
            print(f"Step: {self.steps}, State: {self.state}, Cost: {self.cumulative_cost:.4f}")

    def close(self):
        """Clean up resources."""
        pass


A = np.array([[0., 1.], [0., 0.]])
B = np.eye(2)
Q = np.eye(2)
R = np.eye(2)
Q_f = np.eye(2)

gym.envs.registration.register(
    id='LQR-v0',
    entry_point='lqr_env:LQREnv',
    kwargs={'A': A, 'B': B, 'Q': Q, 'Q_f': Q_f, 'R': R, 'x_bounds': np.array([[-2.0, 2.0], [-2.0, 2.0]])},
    max_episode_steps=200,
)