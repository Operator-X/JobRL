import os
import numpy as np
import gymnasium as gym
from typing import Optional

from config import FactoryConfig
from env import FactoryEnv

# Simple 3x3 instance for training verification
SIMPLE_3x3_JOBS = [
    [(0, 3), (1, 2), (2, 2)],
    [(0, 2), (2, 4), (1, 1)],
    [(1, 4), (0, 3), (2, 3)]
]

class JssRLWrapper(gym.Wrapper):
    """
    Wrapper for standard RL libraries (like Stable-Baselines3).
    Flattens the nested observation dictionary and handles invalid actions 
    by penalizing and re-routing to valid actions (preventing simulation crashes).
    """
    def __init__(self, env: FactoryEnv, invalid_action_penalty: float = -10.0):
        super().__init__(env)
        self.invalid_action_penalty = invalid_action_penalty
        
        # Calculate flat observation dimension
        # 1. Action mask: size of action space
        self.mask_dim = env.observation_space["action_mask"].shape[0]
        # 2. Jobs features: num_jobs * 5
        self.jobs_dim = env.num_jobs * 5
        # 3. Machine features: num_machines * 4
        self.machines_dim = env.num_machines * 4
        
        self.total_obs_dim = self.mask_dim + self.jobs_dim + self.machines_dim
        
        # Override observation space to be flat Box
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.total_obs_dim,),
            dtype=np.float32
        )

    def _flatten_obs(self, obs: dict) -> np.ndarray:
        mask = obs["action_mask"].astype(np.float32)
        jobs = obs["real_obs"]["jobs"].flatten()
        machines = obs["real_obs"]["machines"].flatten()
        return np.concatenate([mask, jobs, machines])

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_action_mask = obs["action_mask"]
        return self._flatten_obs(obs), info

    def step(self, action: int):
        # Check if the chosen action is valid according to the mask
        if self.current_action_mask[action] == 0:
            # Action is invalid! Apply penalty and choose a random valid action to prevent crash
            valid_actions = np.where(self.current_action_mask == 1)[0]
            if len(valid_actions) > 0:
                action = int(np.random.choice(valid_actions))
                penalty = self.invalid_action_penalty
            else:
                # No valid actions (deadlock or completed) - terminate
                obs = self._flatten_obs(self.env._get_obs())
                return obs, self.invalid_action_penalty, True, False, self.env._get_info()
        else:
            penalty = 0.0

        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_action_mask = obs["action_mask"]
        
        # Add penalty to step reward
        total_reward = reward + penalty
        
        return self._flatten_obs(obs), total_reward, terminated, truncated, info


def train_ppo_agent(total_timesteps: int = 2000, model_save_path: str = "ppo_factory_agent"):
    """
    Trains a PPO agent on the JSS environment using Stable-Baselines3.
    """
    # Import Stable-Baselines3 here to avoid crash if not installed when this module is imported
    from stable_baselines3 import PPO
    
    print(f"Initializing FactoryEnv (mode=simplified, 3x3 JSS instance)...")
    config = FactoryConfig(mode="simplified", reward_type="dense_idle_delay")
    raw_env = FactoryEnv(jobs_data=SIMPLE_3x3_JOBS, num_machines=3, config=config)
    env = JssRLWrapper(raw_env, invalid_action_penalty=-5.0)

    print(f"Creating PPO model with MlpPolicy...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=64,
        batch_size=32,
        n_epochs=5,
        gamma=0.99,
        seed=42
    )

    print(f"Training PPO agent for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)

    print(f"Saving trained PPO model to {model_save_path}...")
    model.save(model_save_path)
    print("Training successfully completed!\n")


if __name__ == "__main__":
    # Test execution
    train_ppo_agent(total_timesteps=500)
