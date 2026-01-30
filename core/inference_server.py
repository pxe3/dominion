import torch
import numpy as np
from core.mini_ray.remote import remote

@remote
class InferenceServer:
    def __init__(self, model):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def infer(self, obs):
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            outputs = self.model.forward_all(obs_tensor)
        return {k: v.cpu().numpy() for k, v in outputs.items()}

    def load_weights(self, state_dict):
        self.model.load_state_dict(state_dict)