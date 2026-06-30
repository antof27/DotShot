import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

def make_env_fn(env_id, rank, seed=0, wrapper_class=None, wrapper_kwargs=None):
    def _init():
        import env  # Ensures environment registration in subprocesses
        game_env = gym.make(env_id)
        game_env.reset(seed=seed + rank)
        # Wrap with Monitor to support SB3 statistics logging (resolves user warning)
        game_env = Monitor(game_env)
        if wrapper_class is not None:
            kwargs = wrapper_kwargs if wrapper_kwargs is not None else {}
            game_env = wrapper_class(game_env, **kwargs)
        return game_env
    return _init

def create_vec_env(env_id, num_envs, vec_env_type="dummy", wrapper_class=None, wrapper_kwargs=None, seed=0):
    if vec_env_type == "subproc":
        from stable_baselines3.common.vec_env import SubprocVecEnv
        return SubprocVecEnv([
            make_env_fn(env_id, i, seed=seed, wrapper_class=wrapper_class, wrapper_kwargs=wrapper_kwargs)
            for i in range(num_envs)
        ])
    else:
        from stable_baselines3.common.env_util import make_vec_env
        return make_vec_env(
            env_id,
            n_envs=num_envs,
            wrapper_class=wrapper_class,
            wrapper_kwargs=wrapper_kwargs,
            seed=seed
        )
