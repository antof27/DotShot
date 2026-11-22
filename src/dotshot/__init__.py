"""DotShot Game v2 package: registers Gymnasium Stages 1-2 and exports PettingZoo Stages 3-4."""

from gymnasium.envs.registration import register

from dotshot.envs.single_agent import DotShotSingleEnv
from dotshot.envs.multi_agent import DotShotParallelEnv

for _stage in (1, 2):
    _env_id = f"DotShot-Stage{_stage}-v1"
    register(
        id=_env_id,
        entry_point="dotshot.envs.single_agent:DotShotSingleEnv",
        kwargs={"stage": _stage},
    )

__all__ = ["DotShotSingleEnv", "DotShotParallelEnv"]
