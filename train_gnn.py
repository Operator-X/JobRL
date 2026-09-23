import os
import sys
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim

from config import FactoryConfig
from gnn_policy import GraphActorCritic
from variable_env import VariableJssEnv
from optimizer import JssOptimizer

# 7 Jobs, 6 Machines custom benchmark instance
BENCHMARK_7x6_JOBS = [
    [(0, 3), (1, 2), (2, 2), (3, 4), (4, 1), (5, 3)],
    [(1, 2), (0, 4), (3, 3), (2, 2), (5, 1), (4, 5)],
    [(2, 4), (3, 3), (1, 2), (0, 5), (4, 3), (5, 2)],
    [(0, 3), (2, 2), (4, 4), (1, 1), (3, 2), (5, 4)],
    [(3, 5), (1, 1), (0, 3), (5, 2), (2, 4), (4, 2)],
    [(4, 2), (5, 3), (2, 1), (3, 4), (0, 2), (1, 5)],
    [(5, 4), (4, 2), (3, 3), (2, 1), (1, 3), (0, 4)]
]

# 5 Jobs, 6 Machines benchmark instance
BENCHMARK_5x6_JOBS = BENCHMARK_7x6_JOBS[:5]

# 4 Jobs, 10 Machines benchmark instance
BENCHMARK_4x10_JOBS = [
    [(0, 2), (1, 3), (2, 1), (3, 4), (4, 2), (5, 3), (6, 1), (7, 5), (8, 2), (9, 3)],
    [(1, 4), (0, 2), (3, 3), (2, 4), (5, 2), (4, 1), (7, 3), (6, 2), (9, 4), (8, 1)],
    [(2, 3), (3, 1), (1, 4), (0, 2), (6, 5), (5, 2), (4, 3), (8, 1), (7, 4), (9, 2)],
    [(3, 1), (2, 5), (0, 3), (1, 2), (7, 2), (6, 4), (5, 3), (9, 2), (8, 3), (4, 4)]
]


def evaluate_policy_on_instance(
    policy: GraphActorCritic,
    jobs_data: list,
    num_machines: int,
    deterministic: bool = True,
    device: Optional[torch.device] = None
) -> float:
    """Evaluates the policy on a specific fixed instance and returns the makespan."""
    if device is None:
        device = next(policy.parameters()).device
        
    env = VariableJssEnv(
        fixed_jobs_data=jobs_data,
        fixed_num_machines=num_machines,
        config=FactoryConfig(mode="simplified", reward_type="dense_idle_delay")
    )
    obs, info = env.reset()
    policy.eval()

    done = False
    with torch.no_grad():
        while not done:
            action, _, _, _ = policy.get_action_and_value(
                obs["job_feats"].to(device),
                obs["mach_feats"].to(device),
                obs["target_machines"].to(device),
                obs["action_mask"].to(device),
                mode="simplified",
                deterministic=deterministic
            )
            obs, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated

    return float(info["makespan"])


def train_gnn_ppo(
    total_episodes: int = 400,
    rollout_episodes_per_update: int = 8,
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    ppo_epochs: int = 4,
    save_dir: str = "checkpoints_gnn",
    device_name: str = "auto"
):
    """
    Trains the GraphActorCritic policy using PPO across procedurally generated variable instances.
    Supports auto-detecting CUDA (Colab), MPS (Apple Silicon GPU), or CPU.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Auto-detect device
    if device_name == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_name)

    print(f"Using compute device: {device}")

    # Initialize Policy and Optimizer
    policy = GraphActorCritic(
        job_feat_dim=5,
        mach_feat_dim=4,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        ffn_dim=128
    ).to(device)

    optimizer = optim.Adam(policy.parameters(), lr=lr)

    # Initialize Training Environment (Procedural sampling: 3-8 jobs, 3-6 machines)
    env = VariableJssEnv(
        min_jobs=3,
        max_jobs=8,
        min_machines=3,
        max_machines=6,
        config=FactoryConfig(mode="simplified", reward_type="dense_idle_delay")
    )

    # Compute Exact Ground Truth for benchmark instances
    print("Computing OR-Tools CP-SAT ground truth baselines...")
    opt_7x6_makespan, _ = JssOptimizer(7, 6, BENCHMARK_7x6_JOBS).solve(time_limit_seconds=10.0)
    opt_5x6_makespan, _ = JssOptimizer(5, 6, BENCHMARK_5x6_JOBS).solve(time_limit_seconds=10.0)
    opt_4x10_makespan, _ = JssOptimizer(4, 10, BENCHMARK_4x10_JOBS).solve(time_limit_seconds=10.0)
    print(f"  Optimal 7x6 Makespan : {opt_7x6_makespan}")
    print(f"  Optimal 5x6 Makespan : {opt_5x6_makespan}")
    print(f"  Optimal 4x10 Makespan: {opt_4x10_makespan}")

    # Metrics Tracking
    history_rewards = []
    history_makespans = []
    history_7x6_eval = []
    history_policy_loss = []
    history_value_loss = []
    history_updates = []

    best_7x6_makespan = float("inf")
    current_episode = 0
    update_count = 0

    print("\nStarting Size-Agnostic GNN PPO Training...")
    start_time = time.time()

    while current_episode < total_episodes:
        policy.eval()
        trajectories = []
        batch_rewards = []
        batch_makespans = []

        # 1. Collect Rollouts across variable instances
        for _ in range(rollout_episodes_per_update):
            obs, info = env.reset()
            episode_data = []
            ep_reward = 0.0

            done = False
            while not done:
                with torch.no_grad():
                    action, log_prob, entropy, value = policy.get_action_and_value(
                        obs["job_feats"].to(device),
                        obs["mach_feats"].to(device),
                        obs["target_machines"].to(device),
                        obs["action_mask"].to(device),
                        mode="simplified",
                        deterministic=False
                    )

                next_obs, reward, terminated, truncated, info = env.step(action.item())
                done = terminated or truncated

                episode_data.append({
                    "job_feats": obs["job_feats"],
                    "mach_feats": obs["mach_feats"],
                    "target_machines": obs["target_machines"],
                    "action_mask": obs["action_mask"],
                    "action": action.item(),
                    "log_prob": log_prob.item(),
                    "value": value.item(),
                    "reward": reward,
                    "done": float(done)
                })

                obs = next_obs
                ep_reward += reward

            trajectories.append(episode_data)
            batch_rewards.append(ep_reward)
            batch_makespans.append(info["makespan"])
            current_episode += 1

        # 2. Compute GAE Advantages for each trajectory
        all_transitions = []
        for ep in trajectories:
            T = len(ep)
            last_val = 0.0  # Terminal state value is 0
            running_adv = 0.0

            for t in reversed(range(T)):
                trans = ep[t]
                next_val = ep[t + 1]["value"] if t + 1 < T else last_val
                delta = trans["reward"] + gamma * next_val * (1.0 - trans["done"]) - trans["value"]
                running_adv = delta + gamma * gae_lambda * (1.0 - trans["done"]) * running_adv
                ret = running_adv + trans["value"]

                trans["advantage"] = running_adv
                trans["return"] = ret
                all_transitions.append(trans)

        # 3. PPO Update Epochs
        policy.train()
        advantages = np.array([t["advantage"] for t in all_transitions], dtype=np.float32)
        adv_mean = np.mean(advantages)
        adv_std = np.std(advantages) + 1e-8
        for t in all_transitions:
            t["norm_adv"] = (t["advantage"] - adv_mean) / adv_std

        total_p_loss = 0.0
        total_v_loss = 0.0
        n_updates = 0

        for _ in range(ppo_epochs):
            # Shuffle transitions
            random.shuffle(all_transitions)

            for trans in all_transitions:
                act_tensor = torch.tensor([trans["action"]], dtype=torch.long, device=device)
                adv_tensor = torch.tensor([trans["norm_adv"]], dtype=torch.float32, device=device)
                ret_tensor = torch.tensor([[trans["return"]]], dtype=torch.float32, device=device)
                old_log_prob = trans["log_prob"]

                new_log_prob, entropy, new_val = policy.evaluate_actions(
                    trans["job_feats"].to(device),
                    trans["mach_feats"].to(device),
                    trans["target_machines"].to(device),
                    trans["action_mask"].to(device),
                    act_tensor,
                    mode="simplified"
                )

                # Policy Loss (PPO clipped)
                ratio = torch.exp(new_log_prob - old_log_prob)
                surr1 = ratio * adv_tensor
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_tensor
                policy_loss = - torch.min(surr1, surr2).mean()

                # Value Loss
                value_loss = 0.5 * F.mse_loss(new_val, ret_tensor)

                # Total Loss
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
                optimizer.step()

                total_p_loss += policy_loss.item()
                total_v_loss += value_loss.item()
                n_updates += 1

        update_count += 1
        avg_rew = float(np.mean(batch_rewards))
        avg_mks = float(np.mean(batch_makespans))
        avg_ploss = total_p_loss / max(1, n_updates)
        avg_vloss = total_v_loss / max(1, n_updates)

        # Zero-shot evaluation on 7x6 benchmark
        eval_7x6_mks = evaluate_policy_on_instance(policy, BENCHMARK_7x6_JOBS, 6, deterministic=True)
        if eval_7x6_mks < best_7x6_makespan:
            best_7x6_makespan = eval_7x6_mks
            torch.save(policy.state_dict(), os.path.join(save_dir, "best_gnn_model.pt"))

        history_updates.append(update_count)
        history_rewards.append(avg_rew)
        history_makespans.append(avg_mks)
        history_7x6_eval.append(eval_7x6_mks)
        history_policy_loss.append(avg_ploss)
        history_value_loss.append(avg_vloss)

        if update_count % 5 == 0 or current_episode >= total_episodes:
            print(f"Update {update_count:3d} (Ep {current_episode:3d}/{total_episodes}) | "
                  f"Avg Rew: {avg_rew:7.1f} | Avg Mks: {avg_mks:5.1f} | "
                  f"7x6 Benchmark: {eval_7x6_mks:4.1f} (Best: {best_7x6_makespan:4.1f}, Opt: {opt_7x6_makespan}) | "
                  f"Loss(P/V): {avg_ploss:5.3f}/{avg_vloss:5.3f}")

    # Save latest model
    torch.save(policy.state_dict(), os.path.join(save_dir, "latest_gnn_model.pt"))
    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f}s.")

    # 4. Generate Training Curves Plot
    plot_training_curves(
        history_updates,
        history_rewards,
        history_makespans,
        history_7x6_eval,
        history_policy_loss,
        history_value_loss,
        opt_7x6_makespan,
        save_path="gnn_training_curves.png"
    )

    return policy


def plot_training_curves(
    updates,
    rewards,
    makespans,
    eval_7x6,
    p_loss,
    v_loss,
    opt_7x6,
    save_path="gnn_training_curves.png"
):
    """Generates visual performance graphs for GNN training."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Size-Agnostic GNN / Attention PPO Training Dynamics", fontsize=16, fontweight="bold")

    # Plot 1: Average Episode Reward
    axes[0, 0].plot(updates, rewards, color="#3B82F6", lw=2, label="Rollout Reward")
    axes[0, 0].set_title("Average Episode Return")
    axes[0, 0].set_xlabel("PPO Updates")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Plot 2: Benchmark 7x6 Makespan vs Optimal
    axes[0, 1].plot(updates, eval_7x6, color="#10B981", lw=2, label="GNN Policy (7x6)")
    if opt_7x6 is not None:
        axes[0, 1].axhline(y=opt_7x6, color="#EF4444", linestyle="--", lw=2, label=f"CP-SAT Optimal ({opt_7x6:.0f})")
    axes[0, 1].set_title("Zero-Shot 7x6 Benchmark Makespan")
    axes[0, 1].set_xlabel("PPO Updates")
    axes[0, 1].set_ylabel("Makespan (lower is better)")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Plot 3: Procedural Rollout Average Makespan
    axes[1, 0].plot(updates, makespans, color="#8B5CF6", lw=2, label="Procedural Makespan")
    axes[1, 0].set_title("Average Makespan Across Variable Instances")
    axes[1, 0].set_xlabel("PPO Updates")
    axes[1, 0].set_ylabel("Makespan")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # Plot 4: Losses
    axes[1, 1].plot(updates, p_loss, color="#F59E0B", lw=2, label="Policy Loss")
    axes[1, 1].plot(updates, v_loss, color="#EC4899", lw=2, label="Value Loss")
    axes[1, 1].set_title("Loss Dynamics")
    axes[1, 1].set_xlabel("PPO Updates")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Training curves saved to: {save_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=150, help="Total episodes to train")
    parser.add_argument("--rollout-size", type=int, default=6, help="Episodes per PPO update")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Compute device (auto, cpu, cuda, mps)")
    args = parser.parse_args()

    train_gnn_ppo(
        total_episodes=args.episodes,
        rollout_episodes_per_update=args.rollout_size,
        device_name=args.device
    )
