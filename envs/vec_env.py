import numpy as np
from multiprocessing import Process, Pipe
from typing import Callable, List, Dict, Tuple
from envs.base import BaseEnv


def _worker_fn(conn, env_fn):
    """Process target: runs a single env, responds to commands over a Pipe.

    Protocol:
        ("step", action) -> (obs, reward, done, info)
        ("reset", None)  -> obs
        ("close", None)  -> breaks loop
    """
    env = env_fn()
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            obs, reward, done, info = env.step(data)
            conn.send((obs, reward, done, info))
        elif cmd == "reset":
            obs = env.reset()
            conn.send(obs)
        elif cmd == "close":
            conn.close()
            break


class SubprocVecEnv:
    """Vectorized environment using one subprocess per env.

    Each env.step() runs in parallel across processes. Communication
    happens over multiprocessing Pipes (faster than Queues for 1:1).

    Drop-in replacement for the old sequential VecEnv.
    """

    def __init__(self, env_fn: Callable[[], BaseEnv], num_envs: int):
        self.num_envs = num_envs
        self.parent_conns = []
        self.child_conns = []
        self.procs = []

        for _ in range(num_envs):
            parent_conn, child_conn = Pipe()
            proc = Process(target=_worker_fn, args=(child_conn, env_fn))
            proc.start()
            child_conn.close()  # Parent doesn't use the child end
            self.parent_conns.append(parent_conn)
            self.procs.append(proc)

        # Get shapes from first env (create a throwaway instance)
        dummy = env_fn()
        self.obs_shape = dummy.observation_shape
        self.action_shape = dummy.action_shape

    def reset(self) -> np.ndarray:
        """Reset all envs in parallel, return stacked obs."""
        for conn in self.parent_conns:
            conn.send(("reset", None))
        obs_list = [conn.recv() for conn in self.parent_conns]
        return np.stack(obs_list)

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Step all envs in parallel with the given actions.

        Args:
            actions: (num_envs, action_dim) array.

        Returns:
            obs: (num_envs, obs_dim) array.
            rewards: (num_envs,) array.
            dones: (num_envs,) array.
            infos: List of dicts, one per env.
        """
        # Send all actions at once (non-blocking sends)
        for conn, action in zip(self.parent_conns, actions):
            conn.send(("step", action))

        # Collect all results (blocks until each env finishes)
        results = [conn.recv() for conn in self.parent_conns]

        obs_list, reward_list, done_list, info_list = zip(*results)
        return (
            np.stack(obs_list),
            np.asarray(reward_list),
            np.asarray(done_list),
            list(info_list),
        )

    def close(self):
        """Shut down all env subprocesses."""
        for conn in self.parent_conns:
            conn.send(("close", None))
            conn.close()
        for proc in self.procs:
            proc.join()
