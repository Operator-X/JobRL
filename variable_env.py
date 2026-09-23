import random
import numpy as np
import torch
from typing import List, Tuple, Optional, Dict, Any, Union

from config import FactoryConfig
from engine import JssEngine
from designer_core import generate_random_scenario, to_jobs_data, ScenarioDefinition


class VariableJssEnv:
    """
    Gym-like Environment for Variable-Size Job Shop Scheduling (JSS & FJSP).
    
    Can procedurally resample new factory topologies and problem sizes (N jobs, M machines)
    on every reset(), or evaluate on fixed benchmark instances.
    Returns PyTorch tensors directly formatted for Graph / Attention policies.
    """
    def __init__(
        self,
        min_jobs: int = 3,
        max_jobs: int = 8,
        min_machines: int = 3,
        max_machines: int = 6,
        mode: str = "simplified",
        config: Optional[FactoryConfig] = None,
        fixed_jobs_data: Optional[List[Any]] = None,
        fixed_num_machines: Optional[int] = None
    ):
        self.min_jobs = min_jobs
        self.max_jobs = max_jobs
        self.min_machines = min_machines
        self.max_machines = max_machines
        self.mode = mode
        self.config = config if config is not None else FactoryConfig(mode=mode)
        
        self.fixed_jobs_data = fixed_jobs_data
        self.fixed_num_machines = fixed_num_machines
        
        # Engine placeholder
        self.engine: Optional[JssEngine] = None
        self.num_jobs = 0
        self.num_machines = 0
        self.jobs_data: List[Any] = []
        
        # Initialize first episode
        self.reset()

    def set_fixed_instance(self, jobs_data: List[Any], num_machines: int):
        """Switches the environment to evaluate a fixed benchmark instance."""
        self.fixed_jobs_data = jobs_data
        self.fixed_num_machines = num_machines

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Resets the environment.
        If fixed_jobs_data is configured, reloads that instance.
        Otherwise, procedurally samples random N and M and generates a new instance.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            self.config.seed = seed

        # Check options for dynamic override
        if options and "fixed_jobs_data" in options:
            self.jobs_data = options["fixed_jobs_data"]
            self.num_jobs = len(self.jobs_data)
            self.num_machines = options.get("num_machines", self.fixed_num_machines)
        elif self.fixed_jobs_data is not None and self.fixed_num_machines is not None:
            self.jobs_data = self.fixed_jobs_data
            self.num_jobs = len(self.jobs_data)
            self.num_machines = self.fixed_num_machines
        else:
            # Procedural Generation
            self.num_jobs = random.randint(self.min_jobs, self.max_jobs)
            self.num_machines = random.randint(self.min_machines, self.max_machines)
            scenario = generate_random_scenario(
                num_jobs=self.num_jobs,
                num_machines=self.num_machines,
                mode=self.mode,
                min_dur=1.0,
                max_dur=5.0
            )
            self.jobs_data = to_jobs_data(scenario)

        # Initialize Simulation Engine
        self.engine = JssEngine(
            num_jobs=self.num_jobs,
            num_machines=self.num_machines,
            jobs_data=self.jobs_data,
            config=self.config
        )

        # Fast forward if no initial actions are legal
        mask = self.engine.get_action_mask()
        if not np.any(mask) and not self.engine.is_all_completed:
            self.engine.fast_forward()

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Executes a dispatch action, advances time until next decision point,
        and computes reward.
        """
        obs = self._get_obs()
        action_mask = obs["action_mask"].numpy()

        if action < 0 or action >= len(action_mask) or action_mask[action] == 0:
            raise ValueError(f"Action {action} is invalid. Action mask: {action_mask}")

        # Dispatch job on machine
        if self.config.mode == "simplified":
            job_id = action
            machine_id = self.engine.jobs[job_id].current_op_options[0][0]
        else:
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
        terminated = self.engine.is_all_completed
        truncated = False

        next_obs = self._get_obs()
        info = self._get_info()
        return next_obs, reward, terminated, truncated, info

    def _get_obs(self) -> Dict[str, Any]:
        """Builds size-agnostic PyTorch tensor representations."""
        raw_mask = self.engine.get_action_mask()
        T_max = max(1.0, float(self.engine.total_duration_sum))
        max_p = max(1.0, float(self.engine.max_op_duration))

        # Job Features: [N, 5]
        jobs_feats = []
        target_machines = []
        for job in self.engine.jobs:
            if job.is_completed:
                completed_ratio = 1.0
                remaining_work = 0.0
                curr_op_duration = 0.0
                next_machine_norm = 0.0
                target_m = 0
            else:
                completed_ratio = job.current_op_idx / max(1, len(job.operations))
                remaining_work = sum(
                    min(opt[1] for opt in op_opts) for op_opts in job.operations[job.current_op_idx:]
                ) / T_max
                curr_options = job.current_op_options
                curr_op_duration = min(opt[1] for opt in curr_options) / max_p
                target_m = curr_options[0][0]
                next_machine_norm = target_m / max(1, self.num_machines)

            wait_time = (self.engine.current_time - job.last_action_time) / T_max

            jobs_feats.append([
                completed_ratio,
                remaining_work,
                curr_op_duration,
                next_machine_norm,
                wait_time
            ])
            target_machines.append(min(self.num_machines - 1, max(0, int(target_m))))

        # Machine Features: [M, 4]
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
            "job_feats": torch.tensor(jobs_feats, dtype=torch.float32),
            "mach_feats": torch.tensor(machines_feats, dtype=torch.float32),
            "target_machines": torch.tensor(target_machines, dtype=torch.long),
            "action_mask": torch.tensor(raw_mask, dtype=torch.float32),
            "num_jobs": self.num_jobs,
            "num_machines": self.num_machines
        }

    def _get_info(self) -> Dict[str, Any]:
        return {
            "makespan": self.engine.current_time,
            "completed_jobs": [job.job_id for job in self.engine.jobs if job.is_completed],
            "num_jobs": self.num_jobs,
            "num_machines": self.num_machines
        }

    def _compute_reward(self, dt: float, machine_idle_times: Dict[int, float]) -> float:
        if self.config.reward_type == "dense_idle_delay":
            total_idle_time = sum(machine_idle_times.values())
            gamma = self.config.idle_weight_gamma
            return - (dt + gamma * total_idle_time)
        elif self.config.reward_type == "sparse_makespan":
            if self.engine.is_all_completed:
                return - self.engine.current_time
            return 0.0
        return 0.0
