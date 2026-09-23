import torch
import numpy as np

from gnn_policy import GraphActorCritic
from variable_env import VariableJssEnv


def test_gnn_policy_variable_dimensions():
    """Verify forward pass on diverse instance dimensions."""
    policy = GraphActorCritic(
        job_feat_dim=5,
        mach_feat_dim=4,
        embed_dim=32,
        num_heads=2,
        num_layers=1,
        ffn_dim=64
    )

    test_cases = [
        (5, 6),   # 5 jobs, 6 machines
        (4, 10),  # 4 jobs, 10 machines
        (7, 6),   # 7 jobs, 6 machines (custom benchmark size)
        (3, 3),   # 3 jobs, 3 machines (mini size)
        (12, 8),  # 12 jobs, 8 machines (larger size)
    ]

    for n_jobs, n_machs in test_cases:
        job_feats = torch.randn(n_jobs, 5)
        mach_feats = torch.randn(n_machs, 4)
        target_machines = torch.randint(0, n_machs, (n_jobs,))
        action_mask = torch.ones(n_jobs)
        action_mask[0] = 0  # mask first job

        # 1. Action and Value
        action, log_prob, entropy, value = policy.get_action_and_value(
            job_feats,
            mach_feats,
            target_machines,
            action_mask,
            mode="simplified",
            deterministic=False
        )

        assert action.shape == torch.Size([1])
        assert action.item() != 0, "Action 0 is masked out and must never be selected!"
        assert log_prob.shape == torch.Size([1])
        assert value.shape == torch.Size([1, 1])

        # 2. Evaluation
        actions = torch.tensor([action.item()])
        eval_log_prob, eval_entropy, eval_value = policy.evaluate_actions(
            job_feats,
            mach_feats,
            target_machines,
            action_mask,
            actions,
            mode="simplified"
        )
        assert torch.allclose(log_prob, eval_log_prob, atol=1e-5)
        assert torch.allclose(value, eval_value, atol=1e-5)


def test_critic_permutation_invariance():
    """Verify that Critic value V(s) is invariant to job order permutation."""
    policy = GraphActorCritic(
        job_feat_dim=5,
        mach_feat_dim=4,
        embed_dim=32,
        num_heads=2,
        num_layers=1
    )
    policy.eval()

    n_jobs, n_machs = 5, 4
    job_feats = torch.randn(n_jobs, 5)
    mach_feats = torch.randn(n_machs, 4)

    with torch.no_grad():
        h_jobs1, h_machs1 = policy.encode(job_feats, mach_feats)
        v1 = policy.compute_value(h_jobs1, h_machs1)

        # Permute jobs
        perm = torch.randperm(n_jobs)
        job_feats_perm = job_feats[perm]

        h_jobs2, h_machs2 = policy.encode(job_feats_perm, mach_feats)
        v2 = policy.compute_value(h_jobs2, h_machs2)

    # Invariant to permutation: V(s) must be identical
    assert torch.allclose(v1, v2, atol=1e-4), f"Critic value changed under permutation: {v1.item()} vs {v2.item()}"


def test_actor_masking():
    """Verify invalid actions are strictly masked to -1e9."""
    policy = GraphActorCritic(job_feat_dim=5, mach_feat_dim=4, embed_dim=32)
    job_feats = torch.randn(4, 5)
    mach_feats = torch.randn(3, 4)
    target_machines = torch.tensor([0, 1, 2, 0])
    action_mask = torch.tensor([1.0, 0.0, 0.0, 1.0])  # Only 0 and 3 are valid

    h_jobs, h_machs = policy.encode(job_feats, mach_feats)
    logits = policy.compute_logits(h_jobs, h_machs, target_machines, action_mask)

    assert logits[0, 1].item() < -1e8
    assert logits[0, 2].item() < -1e8
    assert logits[0, 0].item() > -1e4
    assert logits[0, 3].item() > -1e4


def test_variable_env_integration():
    """Test stepping through VariableJssEnv with procedural generation."""
    env = VariableJssEnv(min_jobs=3, max_jobs=6, min_machines=3, max_machines=5)
    obs, info = env.reset()

    assert obs["job_feats"].ndim == 2
    assert obs["mach_feats"].ndim == 2
    assert obs["action_mask"].ndim == 1
    assert len(obs["action_mask"]) == env.num_jobs

    done = False
    step_count = 0
    while not done and step_count < 100:
        valid_actions = torch.where(obs["action_mask"] == 1.0)[0]
        if len(valid_actions) == 0:
            break
        act = int(valid_actions[0].item())
        obs, reward, terminated, truncated, info = env.step(act)
        done = terminated or truncated
        step_count += 1

    assert step_count > 0
    assert "makespan" in info


if __name__ == "__main__":
    print("Running test_gnn_policy_variable_dimensions()...")
    test_gnn_policy_variable_dimensions()
    print("Running test_critic_permutation_invariance()...")
    test_critic_permutation_invariance()
    print("Running test_actor_masking()...")
    test_actor_masking()
    print("Running test_variable_env_integration()...")
    test_variable_env_integration()
    print("\n✅ ALL TESTS PASSED SUCCESSFULLY!")
