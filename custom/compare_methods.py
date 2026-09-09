import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

# Add parent directory to sys.path to allow imports of parent modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import FactoryConfig
from env import FactoryEnv
from train import JssRLWrapper
from optimizer import JssOptimizer

# 7 Jobs, 6 Machines JSS instance configuration
MY_7x6_JOBS = [
    [(0, 3), (1, 2), (2, 2), (3, 4), (4, 1), (5, 3)],
    [(1, 2), (0, 4), (3, 3), (2, 2), (5, 1), (4, 5)],
    [(2, 4), (3, 3), (1, 2), (0, 5), (4, 3), (5, 2)],
    [(0, 3), (2, 2), (4, 4), (1, 1), (3, 2), (5, 4)],
    [(3, 5), (1, 1), (0, 3), (5, 2), (2, 4), (4, 2)],
    [(4, 2), (5, 3), (2, 1), (3, 4), (0, 2), (1, 5)],
    [(5, 4), (4, 2), (3, 3), (2, 1), (1, 3), (0, 4)]
]

# Vibrant colors for the 7 Jobs
COLORS = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899", "#6366F1"]

def plot_gantt(log, title, save_path):
    """
    Helper function to plot a Gantt chart from a simulation or optimization log.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Process log into operations intervals
    ops = {}
    for entry in log:
        job_id = int(entry["job_id"])
        if entry["event"] == "start":
            op_idx = int(entry["op_idx"])
            machine_id = int(entry["machine_id"])
            start_time = float(entry["time"])
            duration = float(entry["duration"])
            ops[(job_id, op_idx)] = {
                "machine_id": machine_id,
                "start": start_time,
                "duration": duration,
                "end": start_time + duration
            }

    for (job_id, op_idx), op in ops.items():
        m_id = op["machine_id"]
        start = op["start"]
        dur = op["duration"]
        
        # Draw bar
        ax.barh(
            y=m_id, 
            width=dur, 
            left=start, 
            height=0.4, 
            color=COLORS[job_id % len(COLORS)],
            edgecolor="black",
            alpha=0.85
        )
        
        # Add text label
        ax.text(
            x=start + dur / 2, 
            y=m_id, 
            s=f"J{job_id}\nO{op_idx}", 
            va="center", 
            ha="center", 
            color="white", 
            fontweight="bold",
            fontsize=8
        )

    # Labels and formatting
    ax.set_yticks(range(6))
    ax.set_yticklabels([f"Machine {i}" for i in range(6)], fontsize=12)
    ax.set_xlabel("Time (Seconds)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.grid(axis="x", linestyle="--", alpha=0.7)
    
    # Create legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=COLORS[i], edgecolor="black", label=f"Job {i}") for i in range(7)]
    ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.15, 1.0), fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_spt_heuristic():
    """
    Runs a rollout on FactoryEnv using the Shortest Processing Time (SPT) rule.
    """
    config = FactoryConfig(mode="simplified")
    raw_env = FactoryEnv(jobs_data=MY_7x6_JOBS, num_machines=6, config=config)
    
    obs, info = raw_env.reset(seed=42)
    terminated = False
    
    while not terminated:
        action_mask = obs["action_mask"]
        valid_actions = np.where(action_mask == 1)[0]
        if len(valid_actions) == 0:
            break
            
        # Select valid job with shortest processing time for its current operation
        best_action = None
        min_duration = float("inf")
        for action in valid_actions:
            job = raw_env.engine.jobs[action]
            duration = job.current_op_options[0][1]
            if duration < min_duration:
                min_duration = duration
                best_action = action
                
        obs, reward, terminated, _, info = raw_env.step(best_action)
        
    return raw_env.engine.log, raw_env.engine.current_time


def run_ppo_rl():
    """
    Trains and runs the PPO Reinforcement Learning agent.
    """
    config = FactoryConfig(mode="simplified", reward_type="dense_idle_delay")
    raw_env = FactoryEnv(jobs_data=MY_7x6_JOBS, num_machines=6, config=config)
    wrapped_env = JssRLWrapper(raw_env, invalid_action_penalty=-5.0)
    env = Monitor(wrapped_env)

    print("Training PPO Agent for 20000 steps...")
    model = PPO("MlpPolicy", env, verbose=0, n_steps=128, batch_size=64, learning_rate=3e-4, seed=42)
    model.learn(total_timesteps=20000)
    
    # Rollout
    obs, info = env.reset(seed=42)
    terminated = False
    while not terminated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, _, info = env.step(action)
        
    return raw_env.engine.log, raw_env.engine.current_time


def run_cpsat_solver():
    """
    Solves JSS using Google OR-Tools CP-SAT exact optimizer.
    """
    optimizer = JssOptimizer(num_jobs=7, num_machines=6, jobs_data=MY_7x6_JOBS)
    makespan, schedule_info = optimizer.solve(time_limit_seconds=10.0)
    
    # Translate schedule_info into JssEngine-like log list so we can use our plotting code
    log = []
    if schedule_info:
        for op_key, val in schedule_info.items():
            # Add start event
            log.append({
                "event": "start",
                "time": val["start"],
                "job_id": val["job_id"],
                "op_idx": val["op_idx"],
                "machine_id": val["machine_id"],
                "duration": val["end"] - val["start"]
            })
            # Add complete event
            log.append({
                "event": "complete",
                "time": val["end"],
                "job_id": val["job_id"],
                "op_idx": val["op_idx"],
                "machine_id": val["machine_id"]
            })
    # Sort events by time to be sequential
    log.sort(key=lambda x: x["time"])
    return log, makespan


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(script_dir, exist_ok=True)
    
    print("==================================================")
    print("1. Running exact solver (Google OR-Tools CP-SAT)...")
    cpsat_log, cpsat_makespan = run_cpsat_solver()
    cpsat_gantt_path = os.path.join(script_dir, "gantt_cpsat.png")
    plot_gantt(cpsat_log, "Optimal Schedule (OR-Tools CP-SAT)", cpsat_gantt_path)
    print(f"CP-SAT Makespan: {cpsat_makespan} seconds")
    
    print("\n2. Running heuristic rule-based solver (SPT)...")
    spt_log, spt_makespan = run_spt_heuristic()
    spt_gantt_path = os.path.join(script_dir, "gantt_spt.png")
    plot_gantt(spt_log, "Heuristic Schedule (Shortest Processing Time - SPT)", spt_gantt_path)
    print(f"SPT Makespan: {spt_makespan} seconds")

    print("\n3. Running Reinforcement Learning solver (PPO)...")
    ppo_log, ppo_makespan = run_ppo_rl()
    ppo_gantt_path = os.path.join(script_dir, "gantt_ppo.png")
    plot_gantt(ppo_log, "RL Schedule (PPO Policy)", ppo_gantt_path)
    print(f"PPO Makespan: {ppo_makespan} seconds")
    
    print("\n==================================================")
    print("            SCHEDULING PERFORMANCE COMPARISON")
    print("==================================================")
    print(f"  Method              Makespan (seconds)")
    print("  --------------------------------------------------")
    print(f"  OR-Tools CP-SAT     {cpsat_makespan:5.1f} (Optimal)")
    print(f"  PPO RL Agent        {ppo_makespan:5.1f}")
    print(f"  SPT Heuristic       {spt_makespan:5.1f}")
    print("==================================================")
    print(f"Gantt charts saved in: {script_dir}/")


if __name__ == "__main__":
    main()
