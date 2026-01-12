from env_def import Cars
from ppo_def import PPO, RolloutBuffer
import torch

max_steps = 10000

sim_env = Cars(max_steps,render=False)

action_dim = sim_env.single_action_space[0]
obs_dim = sim_env.single_observation_space[0]
hidden_dim=128
lr=1e-4
gamma=0.99
gae_disc=0.95
eps_clip=0.2
grad_epochs=10
buffer_size = 16384

ppo_impl = PPO(obs_dim, action_dim, hidden_dim, lr, gamma, gae_disc, eps_clip, grad_epochs)
rollout_buffer = RolloutBuffer(buffer_size, obs_dim, action_dim)


state = sim_env.reset()
curr_steps = 0
total_steps = 1000000

episode_returns = []


while curr_steps < total_steps:
    for _ in range(rollout_buffer.size):
        action, log_prob, value = ppo_impl.select_action(state)
        next_state, reward, done, info = sim_env.step(action)
        rollout_buffer.add(state, action, reward, value, done, log_prob)

        state = next_state
        curr_steps += 1

        if done and 'episode' in info:
            episode_returns.append(info['episode']['r'])

    ppo_impl.update(rollout_buffer)
    if episode_returns:
        avg_return = sum(episode_returns[-10:])/ min(len(episode_returns), 10)
        print(f"Steps: {curr_steps}, Episodes: {len(episode_returns)}, Avg Return (last 10): {avg_return:.2f}")

torch.save(ppo_impl.actor.state_dict(), 'actor.pth')
torch.save(ppo_impl.critic.state_dict(), 'critic.pth')
print("Model saved!")




