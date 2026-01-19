import hydra
from omegaconf import DictConfig
import torch
import numpy as np
import os

from envs.car_env import CarEnv
from envs.vec_env import VecEnv
from algos.ppo_def import PPO
from core.buffer import RolloutBuffer


def make_env(cfg):
    def _init():
        return CarEnv(
            max_steps=cfg.env.max_steps,
            goal=cfg.env.goal,
            render=False
        )
    return _init


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def train(cfg: DictConfig):
    # Seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Create vectorized env
    vec_env = VecEnv(make_env(cfg), cfg.num_envs)
    obs_dim = vec_env.obs_shape[0]
    action_dim = vec_env.action_shape[0]

    # Create algo
    ppo = PPO(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=cfg.algo.hidden_dim,
        lr=cfg.algo.lr,
        gamma=cfg.algo.gamma,
        gae_disc=cfg.algo.gae_lambda,
        eps_clip=cfg.algo.eps_clip,
        grad_epochs=cfg.algo.grad_epochs
    )

    # Create buffer
    buffer = RolloutBuffer(
        num_steps=cfg.num_steps,
        num_envs=cfg.num_envs,
        obs_dim=obs_dim,
        act_dim=action_dim
    )

    # Training loop
    states = vec_env.reset()
    total_steps = 0
    num_updates = 0
    episode_returns = []

    while total_steps < cfg.total_timesteps:
        # Collect rollout
        for step in range(cfg.num_steps):
            actions, log_probs, values = ppo.select_action(states)
            next_states, rewards, dones, infos = vec_env.step(actions)

            buffer.add(states, actions, rewards, values, dones, log_probs)
            states = next_states
            total_steps += cfg.num_envs

            # Track episode returns
            for info in infos:
                if 'episode' in info:
                    episode_returns.append(info['episode']['r'])

        # Update policy
        ppo.update(buffer)
        num_updates += 1

        # Logging
        if num_updates % cfg.log_interval == 0 and episode_returns:
            avg_return = np.mean(episode_returns[-100:])
            print(f"Steps: {total_steps:,} | Updates: {num_updates} | "
                  f"Episodes: {len(episode_returns)} | Avg Return (100): {avg_return:.2f}")

        # Checkpointing
        if total_steps % cfg.checkpoint.save_freq < cfg.num_steps * cfg.num_envs:
            os.makedirs(cfg.checkpoint.save_dir, exist_ok=True)
            torch.save(ppo.actor.state_dict(),
                       f"{cfg.checkpoint.save_dir}/actor_{total_steps}.pth")
            torch.save(ppo.critic.state_dict(),
                       f"{cfg.checkpoint.save_dir}/critic_{total_steps}.pth")
            print(f"Checkpoint saved at {total_steps} steps")

    # Final save
    os.makedirs(cfg.checkpoint.save_dir, exist_ok=True)
    torch.save(ppo.actor.state_dict(), f"{cfg.checkpoint.save_dir}/actor_final.pth")
    torch.save(ppo.critic.state_dict(), f"{cfg.checkpoint.save_dir}/critic_final.pth")
    print("Training complete!")


if __name__ == "__main__":
    train()
