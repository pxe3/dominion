# eval.py
from envs.env_def import Cars
from algos.ppo_def import Actor
import torch
import time

print("Starting eval...")

# create env with rendering
env = Cars(max_steps=10000, render=True)
print("Environment created")

# load trained actor
obs_dim = env.single_observation_space[0]
action_dim = env.single_action_space[0]
actor = Actor(obs_dim, action_dim, hidden_dim=128)
actor.load_state_dict(torch.load('actor.pth'))
actor.eval()
print("Actor loaded")

input("Press Enter to start episodes...")

for ep in range(10):

    state = env.reset()
    done = False
    total_reward = 0
    step_count = 0

    while not done:
        state_tensor = torch.FloatTensor(state)
        with torch.no_grad():
            mean, std = actor(state_tensor)
        action = mean.numpy()
        
        state, reward, done, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        env.render()
    
    print(f"Episode {ep+1}, Steps: {step_count}, Return: {total_reward:.2f}")

input("Press Enter to close...")
env.close()