from gym.envs.registration import register

register(
    id='pacmen-tini-4ag-v0',
    entry_point='pac_men.envs:CustomEnv',
    kwargs={
        "n_agents": 4,
    },
)