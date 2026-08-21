import os
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from config import FactoryConfig
from env import FactoryEnv
from train import JssRLWrapper, SIMPLE_3x3_JOBS

def run_agent_and_plot_gantt(model_path: str = "ppo_factory_agent", save_image_path: str = "gantt_chart.png"):
    # Initialize environment
    config = FactoryConfig(mode="simplified", reward_type="dense_idle_delay")
    raw_env = FactoryEnv(jobs_data=SIMPLE_3x3_JOBS, num_machines=3, config=config)
    env = JssRLWrapper(raw_env, invalid_action_penalty=-5.0)

    # Train PPO agent
    print("Training PPO Agent for 2000 steps...")
    model = PPO("MlpPolicy", env, verbose=0, n_steps=64, batch_size=32, seed=42)
    model.learn(total_timesteps=2000)

    # Rollout to generate schedule
    print("Running rollout of the trained agent...")
    obs, info = env.reset(seed=42)
    terminated = False
    
    while not terminated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, _, info = env.step(action)

    # Extract log from raw engine
    log = raw_env.engine.log
    
    # Process log into operations intervals
    # Key: (job_id, op_idx) -> Dict
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

    # Plot Gantt Chart
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Define colors for Jobs
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # Vibrant colors for Job 0, 1, 2
    
    for (job_id, op_idx), op in ops.items():
        m_id = op["machine_id"]
        start = op["start"]
        dur = op["duration"]
        
        # Draw bar
        bar = ax.barh(
            y=m_id, 
            width=dur, 
            left=start, 
            height=0.4, 
            color=colors[job_id % len(colors)],
            edgecolor="black",
            alpha=0.85
        )
        
        # Add text label in the middle of the bar
        ax.text(
            x=start + dur / 2, 
            y=m_id, 
            s=f"J{job_id} O{op_idx}", 
            va="center", 
            ha="center", 
            color="white", 
            fontweight="bold",
            fontsize=10
        )

    # Labels and formatting
    ax.set_yticks(range(3))
    ax.set_yticklabels([f"Machine {i}" for i in range(3)], fontsize=12)
    ax.set_xlabel("Time (Seconds)", fontsize=12)
    ax.set_title("Factory floor Gantt Chart - PPO Scheduler", fontsize=14, fontweight="bold", pad=15)
    ax.grid(axis="x", linestyle="--", alpha=0.7)
    
    # Create legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], edgecolor="black", label=f"Job {i}") for i in range(3)]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)
    
    plt.tight_layout()
    
    # Save chart
    os.makedirs(os.path.dirname(save_image_path), exist_ok=True)
    plt.savefig(save_image_path, dpi=300)
    print(f"Gantt chart successfully saved to {save_image_path}")
    plt.close()

if __name__ == "__main__":
    # Save path in artifacts directory
    save_path = "/Users/lavlinjaison/.gemini/antigravity-ide/brain/3251c115-ca86-4771-8b4b-c48e5d9ed841/gantt_chart.png"
    run_agent_and_plot_gantt(save_image_path=save_path)
