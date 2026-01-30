import numpy as np

class Learner:
    def __init__(self, policy, trajectory_queue, weight_queue):

        self.policy = policy
        self.trajectory_queue = trajectory_queue
        self.weight_queue = weight_queue
        self.update_count = 0

    def run(self):
        while True:
            sample = self.trajectory_queue.get()
            metrics = self.policy.update(sample)
            self.weight_queue.put(self.policy.model.state_dict())
            self.update_count += 1
            if self.update_count % 5 == 0:
                print(f"[Learner] Update {self.update_count} complete")
            