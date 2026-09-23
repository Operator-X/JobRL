import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import List, Dict, Any, Tuple

from config import FactoryConfig
from variable_env import VariableJssEnv
from gnn_policy import GraphActorCritic
from optimizer import JssOptimizer
from designer_core import generate_random_scenario, to_jobs_data

# Vibrant palette for jobs
JOB_COLORS = [
    "#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6",
    "#EC4899", "#6366F1", "#14B8A6", "#F97316", "#84CC16"
]


def run_gnn_policy(
    policy: GraphActorCritic,
    jobs_data: list,
    num_machines: int
) -> Tuple[List[Dict[str, Any]], float]:
    """Runs the GNN policy deterministically and collects the event log and makespan."""
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
                obs["job_feats"],
                obs["mach_feats"],
                obs["target_machines"],
                obs["action_mask"],
                mode="simplified",
                deterministic=True
            )
            obs, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated

    return env.engine.log, float(info["makespan"])


def run_heuristic_policy(
    heuristic: str,
    jobs_data: list,
    num_machines: int
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Runs standard dispatch heuristics:
    - 'spt': Shortest Processing Time
    - 'mwkr': Most Work Remaining
    """
    env = VariableJssEnv(
        fixed_jobs_data=jobs_data,
        fixed_num_machines=num_machines,
        config=FactoryConfig(mode="simplified", reward_type="dense_idle_delay")
    )
    obs, info = env.reset()

    done = False
    while not done:
        mask = obs["action_mask"].numpy()
        valid_actions = np.where(mask == 1.0)[0]
        if len(valid_actions) == 0:
            break

        if heuristic == "spt":
            # Shortest processing time for current operation
            best_act = min(
                valid_actions,
                key=lambda j: env.engine.jobs[j].current_op_options[0][1]
            )
        elif heuristic == "mwkr":
            # Most work remaining
            best_act = max(
                valid_actions,
                key=lambda j: sum(
                    opt[1] for op in env.engine.jobs[j].operations[env.engine.jobs[j].current_op_idx:]
                    for opt in op
                )
            )
        else:
            best_act = valid_actions[0]

        obs, reward, terminated, truncated, info = env.step(int(best_act))
        done = terminated or truncated

    return env.engine.log, float(info["makespan"])


def run_cpsat_solver(
    jobs_data: list,
    num_machines: int,
    time_limit: float = 15.0
) -> Tuple[List[Dict[str, Any]], float]:
    """Solves the instance exactly using Google OR-Tools CP-SAT."""
    optimizer = JssOptimizer(len(jobs_data), num_machines, jobs_data)
    makespan, schedule_info = optimizer.solve(time_limit_seconds=time_limit)

    log = []
    if schedule_info:
        for op_key, val in schedule_info.items():
            log.append({
                "event": "start",
                "time": val["start"],
                "job_id": val["job_id"],
                "op_idx": val["op_idx"],
                "machine_id": val["machine_id"],
                "duration": val["end"] - val["start"]
            })
            log.append({
                "event": "complete",
                "time": val["end"],
                "job_id": val["job_id"],
                "op_idx": val["op_idx"],
                "machine_id": val["machine_id"]
            })
        log.sort(key=lambda x: x["time"])

    return log, float(makespan) if makespan is not None else float("inf")


def plot_gantt_chart(
    log: List[Dict[str, Any]],
    num_machines: int,
    title: str,
    save_path: str
):
    """Generates a clean Gantt chart from simulation/solver logs."""
    fig, ax = plt.subplots(figsize=(11, max(4, num_machines * 0.7)))

    ops = {}
    for entry in log:
        if entry["event"] == "start":
            job_id = int(entry["job_id"])
            op_idx = int(entry["op_idx"])
            m_id = int(entry["machine_id"])
            start_t = float(entry["time"])
            dur = float(entry["duration"])
            ops[(job_id, op_idx)] = (m_id, start_t, dur)

    max_t = 0.0
    for (job_id, op_idx), (m_id, start, dur) in ops.items():
        color = JOB_COLORS[job_id % len(JOB_COLORS)]
        ax.barh(
            y=m_id,
            width=dur,
            left=start,
            height=0.6,
            align="center",
            color=color,
            edgecolor="#1E293B",
            linewidth=1.0,
            alpha=0.9
        )
        ax.text(
            start + dur / 2.0,
            m_id,
            f"J{job_id}",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=8
        )
        max_t = max(max_t, start + dur)

    ax.set_yticks(range(num_machines))
    ax.set_yticklabels([f"Machine {m}" for m in range(num_machines)], fontweight="semibold")
    ax.set_xlabel("Time (seconds)", fontweight="semibold")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    ax.set_xlim(0, max_t * 1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def benchmark_suite(model_path: str = "checkpoints_gnn/best_gnn_model.pt"):
    """
    Executes a comprehensive comparison across multiple instance sizes.
    """
    # Load GNN Policy
    policy = GraphActorCritic(
        job_feat_dim=5,
        mach_feat_dim=4,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        ffn_dim=128
    )

    if os.path.exists(model_path):
        policy.load_state_dict(torch.load(model_path, map_location="cpu"))
        print(f"Loaded trained GNN policy from: {model_path}")
    else:
        print(f"Warning: Model checkpoint '{model_path}' not found! Using initialized weights.")

    policy.eval()

    # Define Test Benchmark Instances
    from train_gnn import BENCHMARK_7x6_JOBS, BENCHMARK_5x6_JOBS, BENCHMARK_4x10_JOBS

    # Procedural 6x5 instance
    proc_scenario = generate_random_scenario(num_jobs=6, num_machines=5, seed=42)
    bench_6x5_jobs = to_jobs_data(proc_scenario)

    instances = [
        ("Benchmark 7x6", BENCHMARK_7x6_JOBS, 6),
        ("Benchmark 5x6", BENCHMARK_5x6_JOBS, 6),
        ("Benchmark 4x10", BENCHMARK_4x10_JOBS, 10),
        ("Random 6x5", bench_6x5_jobs, 5),
    ]

    out_dir = "benchmark_results_gnn"
    os.makedirs(out_dir, exist_ok=True)

    results = []

    print("\n" + "=" * 70)
    print("      SIZE-AGNOSTIC GNN REINFORCEMENT LEARNING BENCHMARK")
    print("=" * 70)

    for name, jobs, n_machs in instances:
        print(f"\nEvaluating on {name} ({len(jobs)} jobs, {n_machs} machines)...")

        # 1. Exact CP-SAT
        cpsat_log, cpsat_mks = run_cpsat_solver(jobs, n_machs)
        # 2. GNN Policy (Zero-shot)
        gnn_log, gnn_mks = run_gnn_policy(policy, jobs, n_machs)
        # 3. SPT Heuristic
        spt_log, spt_mks = run_heuristic_policy("spt", jobs, n_machs)
        # 4. MWKR Heuristic
        mwkr_log, mwkr_mks = run_heuristic_policy("mwkr", jobs, n_machs)

        # Plot Gantt charts for this instance
        prefix = name.lower().replace(" ", "_")
        plot_gantt_chart(
            cpsat_log, n_machs,
            f"{name} - Optimal Schedule (OR-Tools CP-SAT) | Makespan: {cpsat_mks:.1f}s",
            os.path.join(out_dir, f"{prefix}_cpsat.png")
        )
        plot_gantt_chart(
            gnn_log, n_machs,
            f"{name} - Size-Agnostic GNN Policy | Makespan: {gnn_mks:.1f}s",
            os.path.join(out_dir, f"{prefix}_gnn.png")
        )
        plot_gantt_chart(
            spt_log, n_machs,
            f"{name} - SPT Heuristic | Makespan: {spt_mks:.1f}s",
            os.path.join(out_dir, f"{prefix}_spt.png")
        )

        gnn_gap = ((gnn_mks - cpsat_mks) / cpsat_mks) * 100.0 if cpsat_mks > 0 else 0.0
        spt_gap = ((spt_mks - cpsat_mks) / cpsat_mks) * 100.0 if cpsat_mks > 0 else 0.0
        mwkr_gap = ((mwkr_mks - cpsat_mks) / cpsat_mks) * 100.0 if cpsat_mks > 0 else 0.0

        results.append({
            "instance": name,
            "size": f"{len(jobs)}x{n_machs}",
            "cpsat": cpsat_mks,
            "gnn": gnn_mks,
            "gnn_gap": gnn_gap,
            "spt": spt_mks,
            "spt_gap": spt_gap,
            "mwkr": mwkr_mks,
            "mwkr_gap": mwkr_gap
        })

    # Summary Table
    print("\n" + "=" * 78)
    print(f"{'Instance':<15} | {'Size':<6} | {'CP-SAT':<8} | {'GNN Policy':<12} | {'SPT Rule':<10} | {'MWKR Rule':<10}")
    print("-" * 78)
    for r in results:
        print(f"{r['instance']:<15} | {r['size']:<6} | {r['cpsat']:<8.1f} | "
              f"{r['gnn']:<5.1f} ({r['gnn_gap']:+4.1f}%) | "
              f"{r['spt']:<5.1f} ({r['spt_gap']:+4.1f}%) | "
              f"{r['mwkr']:<5.1f} ({r['mwkr_gap']:+4.1f}%)")
    print("=" * 78)
    print(f"Detailed Gantt charts saved to: {out_dir}/\n")


if __name__ == "__main__":
    benchmark_suite()
