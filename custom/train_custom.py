import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

# Add parent directory to sys.path to allow imports of parent modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import FactoryConfig
from env import FactoryEnv
from train import JssRLWrapper

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

class MetricsCallback(BaseCallback):
    """
    Custom callback to log PPO's loss and episode rewards during training.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.losses = []
        self.loss_steps = []
        self.ep_rewards = []
        self.ep_steps = []

    def _on_step(self) -> bool:
        # Check if an episode finished in any of the environments
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.ep_rewards.append(info["episode"]["r"])
                self.ep_steps.append(self.num_timesteps)

        # Check for loss logged by PPO's train step
        # Note: train/loss is written to logger.name_to_value during optimization
        if "train/loss" in self.logger.name_to_value:
            loss_val = self.logger.name_to_value["train/loss"]
            if not self.loss_steps or self.loss_steps[-1] != self.num_timesteps:
                self.losses.append(loss_val)
                self.loss_steps.append(self.num_timesteps)
        
        return True


def train_and_visualize():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    print("Initializing environment...")
    config = FactoryConfig(mode="simplified", reward_type="dense_idle_delay")
    raw_env = FactoryEnv(jobs_data=MY_7x6_JOBS, num_machines=6, config=config)
    
    # Wrap with our custom RL wrapper first
    wrapped_env = JssRLWrapper(raw_env, invalid_action_penalty=-5.0)
    
    # Wrap with Monitor to track episode statistics (needed for episode rewards in callback)
    env = Monitor(wrapped_env)

    # Initialize callback
    metrics_callback = MetricsCallback()

    print("Training PPO Agent for 20000 steps...")
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=0, 
        n_steps=128, 
        batch_size=64, 
        learning_rate=3e-4, 
        seed=42
    )
    
    model.learn(total_timesteps=20000, callback=metrics_callback)
    print("Training finished!")

    # 1. Plot Training Metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss Curve
    if metrics_callback.losses:
        ax1.plot(metrics_callback.loss_steps, metrics_callback.losses, color="#EF4444", label="Total Loss", linewidth=1.5)
        ax1.set_title("PPO Loss Curve", fontsize=14, fontweight="bold", pad=10)
        ax1.set_xlabel("Timesteps", fontsize=12)
        ax1.set_ylabel("Loss", fontsize=12)
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend()
    else:
        ax1.text(0.5, 0.5, "No Loss Logged", va="center", ha="center")
        ax1.set_title("PPO Loss Curve (Empty)")

    # Reward Curve
    if metrics_callback.ep_rewards:
        ax2.plot(metrics_callback.ep_steps, metrics_callback.ep_rewards, color="#3B82F6", alpha=0.3, label="Raw Episode Reward")
        # Add rolling window average for smoothing
        window = min(10, len(metrics_callback.ep_rewards))
        if window > 1:
            rolling_avg = np.convolve(metrics_callback.ep_rewards, np.ones(window)/window, mode='valid')
            ax2.plot(metrics_callback.ep_steps[window-1:], rolling_avg, color="#1D4ED8", label=f"Rolling Avg (W={window})", linewidth=2)
        
        ax2.set_title("Episode Reward Curve", fontsize=14, fontweight="bold", pad=10)
        ax2.set_xlabel("Timesteps", fontsize=12)
        ax2.set_ylabel("Reward", fontsize=12)
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, "No Episode Completed", va="center", ha="center")
        ax2.set_title("Episode Reward Curve (Empty)")

    plt.tight_layout()
    curves_path = os.path.join(script_dir, "training_curves.png")
    plt.savefig(curves_path, dpi=300)
    plt.close()
    print(f"Training metrics plot saved as '{curves_path}'")

    # 2. Run Rollout with the trained agent to generate schedule
    print("Running final model rollout for schedule verification...")
    obs, info = env.reset(seed=42)
    terminated = False
    
    while not terminated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, _, info = env.step(action)

    # Extract log from raw engine
    log = raw_env.engine.log
    
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

    # Plot Gantt Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Define nice vibrant colors for the 7 Jobs
    colors = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899", "#6366F1"]
    
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
            color=colors[job_id % len(colors)],
            edgecolor="black",
            alpha=0.85
        )
        
        # Add text label in the middle of the bar
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
    ax.set_title("Factory floor Gantt Chart (7 Jobs, 6 Machines) - PPO", fontsize=14, fontweight="bold", pad=15)
    ax.grid(axis="x", linestyle="--", alpha=0.7)
    
    # Create legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], edgecolor="black", label=f"Job {i}") for i in range(7)]
    ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.15, 1.0), fontsize=10)
    
    plt.tight_layout()
    gantt_path = os.path.join(script_dir, "custom_gantt_chart.png")
    plt.savefig(gantt_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Gantt chart successfully saved to '{gantt_path}'")


if __name__ == "__main__":
    train_and_visualize()
