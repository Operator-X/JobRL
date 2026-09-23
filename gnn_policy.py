import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import List, Tuple, Dict, Optional, Any, Union
import numpy as np


class BipartiteAttentionBlock(nn.Module):
    """
    Bidirectional cross-attention block between Job tokens and Machine tokens,
    followed by residual connections, LayerNorm, and feedforward MLPs.
    """
    def __init__(self, embed_dim: int = 64, num_heads: int = 4, ffn_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Cross-Attention: Jobs query Machines
        self.cross_attn_job = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm_job1 = nn.LayerNorm(embed_dim)
        
        # Feedforward for Jobs
        self.ffn_job = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim)
        )
        self.norm_job2 = nn.LayerNorm(embed_dim)
        
        # Cross-Attention: Machines query Jobs
        self.cross_attn_mach = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm_mach1 = nn.LayerNorm(embed_dim)
        
        # Feedforward for Machines
        self.ffn_mach = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim)
        )
        self.norm_mach2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        h_jobs: torch.Tensor,
        h_machs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h_jobs: [Batch, N_jobs, embed_dim]
            h_machs: [Batch, N_machs, embed_dim]
        Returns:
            updated_jobs: [Batch, N_jobs, embed_dim]
            updated_machs: [Batch, N_machs, embed_dim]
        """
        # 1. Jobs query Machines (Jobs observe availability & workload of machines)
        attn_jobs, _ = self.cross_attn_job(query=h_jobs, key=h_machs, value=h_machs)
        h_jobs = self.norm_job1(h_jobs + attn_jobs)
        h_jobs = self.norm_job2(h_jobs + self.ffn_job(h_jobs))
        
        # 2. Machines query Jobs (Machines observe queue pressure and job priorities)
        attn_machs, _ = self.cross_attn_mach(query=h_machs, key=h_jobs, value=h_jobs)
        h_machs = self.norm_mach1(h_machs + attn_machs)
        h_machs = self.norm_mach2(h_machs + self.ffn_mach(h_machs))
        
        return h_jobs, h_machs


class GraphActorCritic(nn.Module):
    """
    Size-Agnostic Actor-Critic Policy for Job Shop Scheduling.
    
    Compatible with arbitrary numbers of jobs (N) and machines (M).
    - Actor: Computes dispatch compatibility logits between eligible jobs and machines.
    - Critic: Permutation-invariant pooling over all nodes to estimate state value V(s).
    """
    def __init__(
        self,
        job_feat_dim: int = 5,
        mach_feat_dim: int = 4,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        ffn_dim: int = 128
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Initial node projections
        self.job_embed = nn.Sequential(
            nn.Linear(job_feat_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.mach_embed = nn.Sequential(
            nn.Linear(mach_feat_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Stacked Bipartite Attention Blocks
        self.layers = nn.ModuleList([
            BipartiteAttentionBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim
            )
            for _ in range(num_layers)
        ])
        
        # Actor Dispatch Scoring Head: Evaluates (Job, Machine) pair compatibility
        # Input is concatenation of Job embedding and Target Machine embedding -> scalar logit
        self.actor_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )
        
        # Critic Global Value Head: Pools all Job and Machine embeddings
        # Input is [mean(Jobs), max(Jobs), mean(Machs), max(Machs)] -> 4 * embed_dim
        self.critic_head = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def encode(
        self,
        job_feats: torch.Tensor,
        mach_feats: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes node features through the bipartite attention stack.
        
        Args:
            job_feats: [Batch, N, job_feat_dim] or [N, job_feat_dim]
            mach_feats: [Batch, M, mach_feat_dim] or [M, mach_feat_dim]
        Returns:
            h_jobs: [Batch, N, embed_dim]
            h_machs: [Batch, M, embed_dim]
        """
        # Ensure batch dimension
        if job_feats.ndim == 2:
            job_feats = job_feats.unsqueeze(0)
        if mach_feats.ndim == 2:
            mach_feats = mach_feats.unsqueeze(0)
            
        h_jobs = self.job_embed(job_feats)
        h_machs = self.mach_embed(mach_feats)
        
        for layer in self.layers:
            h_jobs, h_machs = layer(h_jobs, h_machs)
            
        return h_jobs, h_machs

    def compute_logits(
        self,
        h_jobs: torch.Tensor,
        h_machs: torch.Tensor,
        target_machines: torch.Tensor,
        action_mask: torch.Tensor,
        mode: str = "simplified"
    ) -> torch.Tensor:
        """
        Computes masked policy logits.
        
        In simplified mode:
            Each action i corresponds to dispatching Job i to target_machines[i].
            Logit i is computed from [h_jobs[:, i, :], h_machs[:, target_machines[i], :]].
            
        In flexible mode:
            Action k = i * M + j corresponds to (Job i, Machine j).
            
        Args:
            h_jobs: [Batch, N, embed_dim]
            h_machs: [Batch, M, embed_dim]
            target_machines: [Batch, N] machine index for each job in simplified mode
            action_mask: [Batch, num_actions] boolean or 0/1 mask
            mode: "simplified" or "flexible"
        Returns:
            masked_logits: [Batch, num_actions] with invalid actions set to -1e9
        """
        batch_size, num_jobs, _ = h_jobs.shape
        num_machs = h_machs.shape[1]
        
        if mode == "simplified":
            # Ensure target_machines has batch dimension
            if target_machines.ndim == 1:
                target_machines = target_machines.unsqueeze(0)
            # Gather machine embeddings corresponding to each job's required machine
            # target_machines is [Batch, N]
            expanded_indices = target_machines.unsqueeze(-1).expand(-1, -1, self.embed_dim)
            # gathered_machs is [Batch, N, embed_dim]
            gathered_machs = torch.gather(h_machs, dim=1, index=expanded_indices)
            
            # Pairwise representation: [Batch, N, 2 * embed_dim]
            pairs = torch.cat([h_jobs, gathered_machs], dim=-1)
            # Raw logits: [Batch, N]
            raw_logits = self.actor_head(pairs).squeeze(-1)
            
        else:  # flexible mode: all (job, machine) pairs
            jobs_exp = h_jobs.unsqueeze(2).expand(-1, -1, num_machs, -1)
            machs_exp = h_machs.unsqueeze(1).expand(-1, num_jobs, -1, -1)
            pairs = torch.cat([jobs_exp, machs_exp], dim=-1)  # [Batch, N, M, 2 * embed_dim]
            raw_logits = self.actor_head(pairs).view(batch_size, num_jobs * num_machs)
            
        # Ensure action_mask has batch dimension
        if action_mask.ndim == 1:
            action_mask = action_mask.unsqueeze(0)
            
        # Apply action mask: mask invalid actions to large negative value
        mask_bool = (action_mask > 0)
        neg_inf = torch.tensor(-1e9, dtype=raw_logits.dtype, device=raw_logits.device)
        masked_logits = torch.where(mask_bool, raw_logits, neg_inf)
        
        return masked_logits

    def compute_value(
        self,
        h_jobs: torch.Tensor,
        h_machs: torch.Tensor
    ) -> torch.Tensor:
        """
        Permutation-invariant global pooling critic.
        Computes V(s) by aggregating mean and max pooled node representations.
        
        Args:
            h_jobs: [Batch, N, embed_dim]
            h_machs: [Batch, M, embed_dim]
        Returns:
            values: [Batch, 1] scalar state value
        """
        # Pool jobs: mean and max across N dimension
        job_mean = torch.mean(h_jobs, dim=1)
        job_max, _ = torch.max(h_jobs, dim=1)
        
        # Pool machines: mean and max across M dimension
        mach_mean = torch.mean(h_machs, dim=1)
        mach_max, _ = torch.max(h_machs, dim=1)
        
        # Global context vector: [Batch, 4 * embed_dim]
        global_repr = torch.cat([job_mean, job_max, mach_mean, mach_max], dim=-1)
        
        # Value estimate
        value = self.critic_head(global_repr)
        return value

    def get_action_and_value(
        self,
        job_feats: torch.Tensor,
        mach_feats: torch.Tensor,
        target_machines: torch.Tensor,
        action_mask: torch.Tensor,
        mode: str = "simplified",
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples an action from the policy and computes the state value.
        
        Returns:
            action: Selected action index [Batch]
            log_prob: Log probability of the action [Batch]
            entropy: Policy entropy [Batch]
            value: State value V(s) [Batch, 1]
        """
        h_jobs, h_machs = self.encode(job_feats, mach_feats)
        logits = self.compute_logits(h_jobs, h_machs, target_machines, action_mask, mode=mode)
        value = self.compute_value(h_jobs, h_machs)
        
        dist = Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()
            
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value

    def evaluate_actions(
        self,
        job_feats: torch.Tensor,
        mach_feats: torch.Tensor,
        target_machines: torch.Tensor,
        action_mask: torch.Tensor,
        actions: torch.Tensor,
        mode: str = "simplified"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluates actions during PPO updates.
        
        Returns:
            log_probs: [Batch]
            entropies: [Batch]
            values: [Batch, 1]
        """
        h_jobs, h_machs = self.encode(job_feats, mach_feats)
        logits = self.compute_logits(h_jobs, h_machs, target_machines, action_mask, mode=mode)
        value = self.compute_value(h_jobs, h_machs)
        
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropies = dist.entropy()
        
        return log_probs, entropies, value
