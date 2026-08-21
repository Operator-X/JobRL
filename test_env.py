import numpy as np
import random
from typing import List, Tuple, Any

from config import FactoryConfig
from env import FactoryEnv
from optimizer import JssOptimizer
from train import JssRLWrapper

# Standard FT06 instance data (6 jobs, 6 machines JSS benchmark)
FT06_JOBS = [
    [(2, 1), (0, 3), (1, 6), (3, 7), (5, 3), (4, 6)],
    [(1, 8), (2, 5), (4, 10), (5, 10), (0, 10), (3, 4)],
    [(2, 5), (3, 4), (5, 8), (0, 9), (1, 1), (4, 7)],
    [(1, 5), (0, 5), (2, 5), (3, 3), (4, 8), (5, 9)],
    [(2, 9), (1, 3), (4, 5), (0, 4), (3, 3), (5, 1)],
    [(1, 3), (3, 3), (5, 9), (0, 10), (4, 4), (2, 1)]
]

# Simple 3x3 instance
SIMPLE_3x3_JOBS = [
    [(0, 3), (1, 2), (2, 2)],
    [(0, 2), (2, 4), (1, 1)],
    [(1, 4), (0, 3), (2, 3)]
]

# Simple FJSP instance (Flexible Job Shop)
# Job 0 Op 0 can run on Machine 0 (dur 5) or Machine 1 (dur 2)
# Job 1 Op 0 can run on Machine 1 (dur 3) or Machine 2 (dur 6)
# Job 2 Op 0 can run on Machine 2 (dur 4) or Machine 0 (dur 1)
SIMPLE_FJSP_JOBS = [
    [[(0, 5), (1, 2)]],
    [[(1, 3), (2, 6)]],
    [[(2, 4), (0, 1)]]
]

def test_spaces():
    print("Running Space Compliance Test...")
    # Simplified JSS
    env_simple = FactoryEnv(jobs_data=SIMPLE_3x3_JOBS, num_machines=3)
    obs_s, info_s = env_simple.reset()
    assert obs_s["action_mask"].shape == (3,)
    assert env_simple.observation_space.contains(obs_s)

    # Flexible JSS
    config_flex = FactoryConfig(mode="flexible")
    env_flex = FactoryEnv(jobs_data=SIMPLE_3x3_JOBS, num_machines=3, config=config_flex)
    obs_f, info_f = env_flex.reset()
    # Action mask should be shape (num_jobs * num_machines) = (9,)
    assert obs_f["action_mask"].shape == (9,)
    assert env_flex.observation_space.contains(obs_f)
    print("Space Compliance Test: PASSED\n")


def test_flexible_mode_routing():
    print("Running Flexible Mode Routing Test...")
    config = FactoryConfig(mode="flexible")
    env = FactoryEnv(jobs_data=SIMPLE_FJSP_JOBS, num_machines=3, config=config)
    
    # Reset
    obs, info = env.reset()
    action_mask = obs["action_mask"]
    
    # Job 0 Op 0: option 1 is Machine 0 (dur 5), option 2 is Machine 1 (dur 2)
    # Action index for (job_0, machine_1) is 0 * 3 + 1 = 1
    # Action index for (job_0, machine_0) is 0 * 3 + 0 = 0
    assert action_mask[0] == 1, "Job 0 should be compatible with Machine 0"
    assert action_mask[1] == 1, "Job 0 should be compatible with Machine 1"
    assert action_mask[2] == 0, "Job 0 should not be compatible with Machine 2"

    # Step: dispatch Job 0 on Machine 1 (action 1)
    obs, reward, terminated, _, info = env.step(1)
    
    # Assert duration is 2.0 (Machine 1 selection)
    start_log = [e for e in env.engine.log if e["event"] == "start" and e["job_id"] == 0][0]
    assert start_log["duration"] == 2.0, f"Expected duration of 2.0, got {start_log['duration']}"
    assert start_log["machine_id"] == 1, f"Expected machine 1, got {start_log['machine_id']}"
    print("Flexible Mode Routing Test: PASSED\n")


def test_stochastic_breakdowns():
    print("Running Stochastic Breakdowns Test...")
    config = FactoryConfig(
        mode="simplified",
        enable_breakdowns=True,
        failure_rate_lambda=0.05,  # moderate failures to prevent completion starvation
        repair_mean_mu=2.0,
        seed=10
    )
    env = FactoryEnv(jobs_data=SIMPLE_3x3_JOBS, num_machines=3, config=config)
    obs, info = env.reset()
    
    # Dispatch Job 0 on Machine 0
    obs, reward, terminated, _, info = env.step(0)
    
    # Run standard Gym environment loop to completion
    terminated = False
    while not terminated:
        mask = obs["action_mask"]
        valid_actions = np.where(mask == 1)[0]
        if len(valid_actions) > 0:
            action = valid_actions[0]
            obs, reward, terminated, _, info = env.step(action)
        else:
            # If no actions are legal, step should have already terminated.
            break

    # Verify that breakdown events are present in logs
    log = env.engine.log
    breakdown_events = [e for e in log if e["event"] == "breakdown"]
    repair_events = [e for e in log if e["event"] == "repair"]
    
    print(f"Breakdowns observed: {len(breakdown_events)}, Repairs observed: {len(repair_events)}")
    assert len(breakdown_events) > 0, "No breakdown events occurred despite high failure rate"
    assert len(repair_events) > 0, "No repair events occurred"
    print("Stochastic Breakdowns Test: PASSED\n")


def test_dynamic_arrivals():
    print("Running Dynamic Arrivals Test...")
    config = FactoryConfig(
        mode="simplified",
        enable_dynamic_arrivals=True,
        arrival_rate_lambda=0.05,
        seed=42
    )
    env = FactoryEnv(jobs_data=SIMPLE_3x3_JOBS, num_machines=3, config=config)
    obs, info = env.reset()
    
    # Verify that Job 1 or Job 2 is not arrived yet and is masked out initially
    job_1_release = env.engine.jobs[1].release_time
    assert job_1_release > 0.0, "Job 1 release time should be positive"
    assert not env.engine.jobs[1].has_arrived, "Job 1 should not have arrived yet at reset"
    assert obs["action_mask"][1] == 0, "Job 1 should be masked out since it has not arrived"

    # Run steps until Job 1 has arrived
    terminated = False
    while not env.engine.jobs[1].has_arrived and not terminated:
        mask = obs["action_mask"]
        valid_actions = np.where(mask == 1)[0]
        if len(valid_actions) > 0:
            action = valid_actions[0]
            obs, reward, terminated, _, info = env.step(action)
        else:
            break

    # Let simulation clock advance and check arrival
    assert env.engine.jobs[1].has_arrived, "Job 1 did not arrive after clock progression"
    arrival_logged = any(e["event"] == "job_arrived" and e["job_id"] == 1 for e in env.engine.log)
    assert arrival_logged, "Job 1 arrival event was not logged as clock advanced"
    print("Dynamic Arrivals Test: PASSED\n")


def test_setup_times():
    print("Running Setup Times Test...")
    config = FactoryConfig(
        mode="simplified",
        enable_setup_times=True
    )
    env = FactoryEnv(jobs_data=SIMPLE_3x3_JOBS, num_machines=3, config=config)
    obs, info = env.reset()
    
    # Step 1: Dispatch Job 0 on Machine 0 (duration 3.0)
    obs, reward, terminated, _, info = env.step(0)
    
    # Step 2: Dispatch Job 2 on Machine 1 to trigger clock fast-forward
    obs, reward, terminated, _, info = env.step(2)
    
    # Step 3: Dispatch Job 1 on Machine 0 (requires setup since last was Job 0)
    obs, reward, terminated, _, info = env.step(1)
    
    log = env.engine.log
    starts_on_m0 = [e for e in log if e["event"] == "start" and e["machine_id"] == 0]
    
    # First dispatch (Job 0 on M0)
    assert starts_on_m0[0]["setup_time"] == 0.0, "First dispatch should have 0.0 setup time"
    # Second dispatch (Job 1 on M0)
    assert starts_on_m0[1]["setup_time"] == 2.0, f"Expected setup time of 2.0, got {starts_on_m0[1]['setup_time']}"
    print("Setup Times Test: PASSED\n")


def test_buffer_limits():
    print("Running Buffer Capacity Limits Test...")
    config = FactoryConfig(
        mode="simplified",
        enable_buffer_limits=True,
        buffer_capacity=1
    )
    env = FactoryEnv(jobs_data=SIMPLE_3x3_JOBS, num_machines=3, config=config)
    obs, info = env.reset()
    
    # Under buffer_capacity=1:
    # - Job 0 (next M1): waiting count for M1 is 1 (Job 2). M1 is full. Job 0 is masked.
    # - Job 1 (next M2): waiting count for M2 is 0. M2 is free. Job 1 is unmasked.
    # - Job 2 (next M0): waiting count for M0 is 2. M0 is full. Job 2 is masked.
    expected_mask = np.array([0, 1, 0], dtype=np.int8)
    assert np.array_equal(obs["action_mask"], expected_mask), f"Expected mask {expected_mask}, got {obs['action_mask']}"
    print("Buffer Capacity Limits Test: PASSED\n")


def test_ortools_solver():
    print("Running OR-Tools Optimizer Test...")
    optimizer = JssOptimizer(num_jobs=6, num_machines=6, jobs_data=FT06_JOBS)
    makespan, schedule = optimizer.solve(time_limit_seconds=5.0)
    
    print(f"OR-Tools optimal makespan for FT06: {makespan}")
    assert makespan is not None
    assert makespan <= 56.0 + 1e-6, f"Optimal makespan of FT06 is 56, solver found {makespan}"
    print("OR-Tools Optimizer Test: PASSED\n")


def test_rl_training():
    print("Running RL Agent Training Wrapper Test...")
    from stable_baselines3 import PPO
    
    config = FactoryConfig(mode="simplified")
    raw_env = FactoryEnv(jobs_data=SIMPLE_3x3_JOBS, num_machines=3, config=config)
    env = JssRLWrapper(raw_env)
    
    # Verify shape
    obs, info = env.reset()
    assert obs.shape == (env.total_obs_dim,)
    
    # Train PPO model for a tiny amount of steps
    model = PPO("MlpPolicy", env, verbose=0, n_steps=64, batch_size=32)
    model.learn(total_timesteps=128)
    print("RL Agent Training Wrapper Test: PASSED\n")


if __name__ == "__main__":
    print("=============================================")
    print("Starting FactoryEnv verification test suite...")
    print("=============================================")
    test_spaces()
    test_flexible_mode_routing()
    test_stochastic_breakdowns()
    test_dynamic_arrivals()
    test_setup_times()
    test_buffer_limits()
    
    # Run library integrations
    test_ortools_solver()
    test_rl_training()
    
    print("=============================================")
    print("All Phase 2, 3, and 4 verification tests passed successfully!")
    print("=============================================")
