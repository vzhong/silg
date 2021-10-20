from pathlib import Path

import gym
import os
from gym.envs import registration

from .gym_wrapper import TDWrapper as Base, TD_ROOT


# register gym environments if not already registered
for split in ("train", "dev", "test"):
    name = f"td_stop_res50_{split}-v0"
    if name not in [env.id for env in gym.envs.registry.all()]:
        registration.register(
            id=name,
            entry_point="silg.envs.touchdown.stop_gym_wrapper:TDStopWrapper",
            kwargs=dict(
                features_path=os.path.join(TD_ROOT, 'pca_10.npz'),
                data_json=str(Path(__file__).parent.joinpath(f"data/{split}.json")),
                feat_type='res50',
                path_lengths=os.path.join(TD_ROOT, 'shortest_paths.npz'),
            )
        )

    name = f"td_stop_segs_{split}-v0"
    if name not in [env.id for env in gym.envs.registry.all()]:
        registration.register(
            id=name,
            entry_point="silg.envs.touchdown.stop_gym_wrapper:TDStopWrapper",
            kwargs=dict(
                features_path=os.path.join(TD_ROOT, 'maj_ds_a10.npz'),
                data_json=str(Path(__file__).parent.joinpath(f"data/{split}.json")),
                feat_type='segs',
                path_lengths=os.path.join(TD_ROOT, 'shortest_paths.npz'),
            )
        )


class TDStopWrapper(Base):
    '''
    Wrapper to make TDNavigator into a gym env.
    '''

    def get_observation_space(self):
        obs = super().get_observation_space()
        obs.update({
            "valid": (self.max_actions, ),  # expand valid action by 1
            "x": (self.max_actions, ),  # expand valid action by 1
        })
        return obs

    def __init__(self, *args, **kwargs):
        self.old_obs = None
        super().__init__(*args, **kwargs)
        # environment settings
        self.max_actions += 1  # stop action is last
        self.action_space = list(range(self.max_actions))

    def convert_to_str(self, obs):
        ref = super().convert_to_str(obs)
        ref['valid'][-1] = 1  # last stop action is valid
        return ref

    def my_step(self, action):
        stop = action == self.max_actions - 1
        self._step_count += 1
        if stop or self._step_count >= self.max_steps:
            done = True
            # stop action
            if self.nav.get_cur_panoid() == self.nav.target_panoid:
                reward = 10  # stopped at correct pano
            elif stop:
                reward = -10  # stopped at wrong pano
            else:
                reward = -1  # stopped because timed out
        else:
            done = False
            x = self._cur_xs[action].item()
            obs = self.nav.step(x)
            self.old_obs = self._reformat(obs)
            # compute reward
            new_dist = self._distance_to_target()
            reward = (self._distance - new_dist)
            reward = reward / 10
            self._distance = new_dist

        # info
        info = {}
        if self.cache_size:
            info['cache_info'] = self._feature_loader.cache_info()
        reward += self.time_penalty
        return self.old_obs, reward, done, info
