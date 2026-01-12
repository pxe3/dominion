# eval.py
from env_def import Cars
from ppo_def import Actor
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
    print(f"Starting episode {ep+1}")
    print(f"State shape: {state.shape}")
    print(f"State: {state}")
    print(f"Actor input shape: {actor.net[0].weight.shape}")

    state_tensor = torch.FloatTensor(state)
    mean, std = actor(state_tensor)
    print(f"Mean action: {mean}")
    print(f"Std: {std}")

    done = False
    total_reward = 0
    step_count = 0
    
    while not done:
        state_tensor = torch.FloatTensor(state)
        action, _ = actor.get_action(state_tensor)
        action = mean.detach().numpy()
        
        state, reward, done, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        env.render()
    
    print(f"Episode {ep+1}, Steps: {step_count}, Return: {total_reward:.2f}")

input("Press Enter to close...")
env.close()