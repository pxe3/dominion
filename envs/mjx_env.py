"""
MJX environment wrapper using Brax.
Used for distributed training infra. Swap in custom MJX envs later.
"""
import jax
import jax.numpy as jnp
import numpy as np
from brax import envs


class MJXEnv:
    """Brax env wrapper - runs on JAX/MJX, outputs numpy for PyTorch."""

    def __init__(self, env_name: str = "ant", num_envs: int = 1, seed: int = 0):
        self.env = envs.get_environment(env_name)
        self.num_envs = num_envs
        self.rng = jax.random.PRNGKey(seed)

        self._observation_shape = (self.env.observation_size,)
        self._action_shape = (self.env.action_size,)

        self._batched_reset = jax.jit(jax.vmap(self.env.reset))
        self._batched_step = jax.jit(jax.vmap(self.env.step))

        self.state = None
        self.reset()

    @property
    def observation_shape(self):
        return self._observation_shape

    @property
    def obs_shape(self):  # alias for VecEnv compatibility
        return self._observation_shape

    @property
    def action_shape(self):
        return self._action_shape

    def reset(self):
        self.rng, *subkeys = jax.random.split(self.rng, self.num_envs + 1)
        self.state = self._batched_reset(jnp.stack(subkeys))
        return np.array(self.state.obs)

    def step(self, actions):
        self.state = self._batched_step(self.state, jnp.array(actions))
        return (
            np.array(self.state.obs),
            np.array(self.state.reward),
            np.array(self.state.done),
            {}
        )


if __name__ == "__main__":
    env = MJXEnv("ant", num_envs=4)
    print(f"obs: {env.observation_shape}, act: {env.action_shape}")
    obs = env.reset()
    for i in range(3):
        obs, rew, done, _ = env.step(np.random.randn(4, 8))
        print(f"step {i}: rew={rew.round(2)}")
