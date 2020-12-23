import gym
import itertools
from pac_men.pac_men import CustomEnv

_sizes = {
    "tiny": (1, 3, 5),
}

gym.register(
    id=f"pacmen-tiny-4ag-v0",
    entry_point="pac_men:CustomEnv",
    kwargs={
        "n_agents": 4,
    },
)


def full_registration():
    gym.register(
        id=f"pacmen-tiny-4ag-v0",
        entry_point="pac_men:CustomEnv",
        kwargs={
            "n_agents": 4,
        },
    )
