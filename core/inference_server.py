import torch


class InferenceServer:
    """Runs algo.predict() in a dedicated process, serving inference requests via Queues.

    Sits between Worker and Learner:
    - Worker sends obs (numpy) through obs_queue, gets back actions (numpy) from action_queue
    - Learner sends updated weights through weight_queue after each training step

    The server owns an algo instance on a specific device. It handles
    numpy<->tensor conversion at the boundary so algos only work with tensors.
    """

    def __init__(self, algo, device, obs_queue, action_queue, weight_queue):
        """
        Args:
            algo: A BaseAlgo instance (e.g. PPO). Must implement predict() and model property.
            device: Device string (e.g. 'cuda:0', 'cpu') to run inference on.
            obs_queue: Queue[np.ndarray] — observations from Worker.
            action_queue: Queue[Dict[str, np.ndarray]] — inference outputs back to Worker.
            weight_queue: Queue[state_dict] — weight updates from Learner.
        """
        self.algo = algo
        self.device = torch.device(device)
        self.algo.model.to(self.device)
        self.obs_queue = obs_queue
        self.action_queue = action_queue
        self.weight_queue = weight_queue

    def run(self):
        """Main loop: load initial weights, then serve inference requests forever.

        Weight updates are checked non-blocking between each inference call.
        This means weights are at most 1 inference call stale after an update arrives.
        """
        # Block until Learner sends initial weights
        initial_weights = self.weight_queue.get()
        self.algo.model.load_state_dict(initial_weights)
        print("[InferenceServer] Loaded initial weights, ready to serve")

        serve_count = 0
        while True:
            # Drain any pending weight updates (non-blocking)
            while not self.weight_queue.empty():
                new_weights = self.weight_queue.get()
                self.algo.model.load_state_dict(new_weights)
                print("[InferenceServer] Loaded updated weights")

            # Block until Worker sends an observation batch
            obs_np = self.obs_queue.get()

            # numpy -> tensor -> predict -> numpy
            obs_tensor = torch.FloatTensor(obs_np).to(self.device)
            outputs = self.algo.predict(obs_tensor)
            outputs_np = {k: v.cpu().numpy() for k, v in outputs.items()}

            self.action_queue.put(outputs_np)

            serve_count += 1
            if serve_count % 500 == 0:
                print(f"[InferenceServer] Served {serve_count} inference requests")
