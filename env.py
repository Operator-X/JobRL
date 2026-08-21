import gymnasium as gym
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

from config import FactoryConfig
from engine import JssEngine

class FactoryEnv(gym.Env):
    """
    Gymnasium environment for Modular Job Shop Scheduling (JSS & FJSP).
    Coordinates the JssEngine with RL agent actions and observation spaces.
    """
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        jobs_data: List[Any],
        num_machines: int,
        config: Optional[FactoryConfig] = None,
        render_mode: Optional[str] = None
    ):
        super().__init__()
        self.jobs_data = jobs_data
        self.num_jobs = len(jobs_data)
        self.num_machines = num_machines
        self.config = config if config is not None else FactoryConfig()
        self.render_mode = render_mode

        self.engine = JssEngine(self.num_jobs, self.num_machines, self.jobs_data, self.config)

        # Dynamic Action & Observation spaces based on mode
        if self.config.mode == "simplified":
            self.action_space = gym.spaces.Discrete(self.num_jobs)
            mask_shape = (self.num_jobs,)
        else:  # flexible mode
            self.action_space = gym.spaces.Discrete(self.num_jobs * self.num_machines)
            mask_shape = (self.num_jobs * self.num_machines,)

        self.observation_space = gym.spaces.Dict({
            "action_mask": gym.spaces.Box(0, 1, shape=mask_shape, dtype=np.int8),
            "real_obs": gym.spaces.Dict({
                "jobs": gym.spaces.Box(0.0, 1.0, shape=(self.num_jobs, 5), dtype=np.float32),
                "machines": gym.spaces.Box(0.0, 1.0, shape=(self.num_machines, 4), dtype=np.float32),
            })
        })

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Resets the environment."""
        super().reset(seed=seed)
        
        if seed is not None:
            self.config.seed = seed
            # Recreate engine to reset RandomState seed
            self.engine = JssEngine(self.num_jobs, self.num_machines, self.jobs_data, self.config)
        else:
            self.engine.reset()

        # Fast forward if no actions are available initially
        mask = self.engine.get_action_mask()
        if not np.any(mask) and not self.engine.is_all_completed:
            self.engine.fast_forward()

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Dispatches the selected job, advances simulation clock until next decision point,
        and computes reward.
        """
        # Validate Action
        obs = self._get_obs()
        action_mask = obs["action_mask"]
        if action_mask[action] == 0:
            raise ValueError(f"Action {action} is invalid. Action mask is: {action_mask}")

        # Dispatch job
        if self.config.mode == "simplified":
            job_id = action
            # Simplified mode always schedules on the first option's machine
            machine_id = self.engine.jobs[job_id].current_op_options[0][0]
        else:  # flexible mode
            job_id = action // self.num_machines
            machine_id = action % self.num_machines

        self.engine.dispatch(job_id, machine_id)

        # Fast forward if no actions are legal
        dt = 0.0
        machine_idle_times = {m.machine_id: 0.0 for m in self.engine.machines}
        
        next_mask = self.engine.get_action_mask()
        if not np.any(next_mask) and not self.engine.is_all_completed:
            dt, machine_idle_times = self.engine.fast_forward()

        # Compute reward
        reward = self._compute_reward(dt, machine_idle_times)

        # Terminated / Truncated status
        terminated = self.engine.is_all_completed
        truncated = False

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> Dict[str, Any]:
        """Constructs and returns the normalized observation dict."""
        action_mask = self.engine.get_action_mask()
        T_max = self.engine.total_duration_sum
        max_p = self.engine.max_op_duration

        # Job Features
        jobs_feats = []
        for job in self.engine.jobs:
            if job.is_completed:
                completed_ratio = 1.0
                remaining_work = 0.0
                curr_op_duration = 0.0
                next_machine_norm = 0.0
            else:
                completed_ratio = job.current_op_idx / len(job.operations)
                # Sum of min processing duration for remaining steps
                remaining_work = sum(
                    min(opt[1] for opt in op_opts) for op_opts in job.operations[job.current_op_idx:]
                ) / T_max
                curr_options = job.current_op_options
                curr_op_duration = min(opt[1] for opt in curr_options) / max_p
                next_machine_norm = curr_options[0][0] / self.num_machines

            wait_time = (self.engine.current_time - job.last_action_time) / T_max

            jobs_feats.append([
                completed_ratio,
                remaining_work,
                curr_op_duration,
                next_machine_norm,
                wait_time
            ])

        # Machine Features
        machines_feats = []
        for machine in self.engine.machines:
            is_busy_val = 1.0 if machine.is_busy else 0.0
            rem_time_val = max(0.0, machine.busy_until - self.engine.current_time) / max_p
            total_scheduled_val = machine.total_work_scheduled / T_max
            is_broken_val = 1.0 if machine.is_broken else 0.0

            machines_feats.append([
                is_busy_val,
                rem_time_val,
                total_scheduled_val,
                is_broken_val
            ])

        return {
            "action_mask": action_mask,
            "real_obs": {
                "jobs": np.array(jobs_feats, dtype=np.float32),
                "machines": np.array(machines_feats, dtype=np.float32),
            }
        }

    def _get_info(self) -> Dict[str, Any]:
        """Constructs the diagnostic info dictionary."""
        return {
            "makespan": self.engine.current_time,
            "completed_jobs": [job.job_id for job in self.engine.jobs if job.is_completed],
        }

    def _compute_reward(self, dt: float, machine_idle_times: Dict[int, float]) -> float:
        """Computes step-level reward according to configuration settings."""
        if self.config.reward_type == "dense_idle_delay":
            total_idle_time = sum(machine_idle_times.values())
            gamma = self.config.idle_weight_gamma
            return - (dt + gamma * total_idle_time)
        elif self.config.reward_type == "sparse_makespan":
            if self.engine.is_all_completed:
                return - self.engine.current_time
            return 0.0
        else:
            raise ValueError(f"Unknown reward type: {self.config.reward_type}")

    def render(self) -> Optional[str]:
        """Renders simulation state to console or returns string representation."""
        lines = [f"--- Time: {self.engine.current_time:.2f} ---"]
        for job in self.engine.jobs:
            status = "Done" if job.is_completed else ("Busy" if job.in_flight else "Waiting")
            lines.append(f"  Job {job.job_id}: step {job.current_op_idx}/{len(job.operations)} ({status})")
        for machine in self.engine.machines:
            status = f"Busy on Job {machine.current_job_id}" if machine.is_busy else "Idle"
            is_broken_str = " (BROKEN)" if machine.is_broken else ""
            lines.append(f"  Machine {machine.machine_id}: {status}{is_broken_str} (busy_until: {machine.busy_until:.2f})")
        
        rendered_str = "\n".join(lines)
        if self.render_mode == "human":
            print(rendered_str)
            return None
        return rendered_str
