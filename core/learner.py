import numpy as np

class Learner:
    def __init__(self, policy, trajectory_queue, weight_queue):

        self.policy = policy
        self.trajectory_queue = trajectory_queue
        self.weight_queue = weight_queue
    
    def run(self):
        while True:
            sample = self.trajectory_queue.get()
            self.policy.update(sample)
            self.weight_queue.put(self.policy.ac.state_dict())
            