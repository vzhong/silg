import os
import gym
import torch
import shutil
import numpy as np
from gym.envs import registration
from nle.env.tasks import NetHackGold as NetHackGoldBase, NetHackScout as NetHackScoutBase, NetHackScore as NetHackScoreBase, NetHackEat as NetHackEatBase
from nle.nethack import MAX_GLYPH
from nle import nethack
from .base import SILGEnv


registration.register(
    id="nethack_train-v0", entry_point="silg.envs.nethack:NetHackMultitask"
)
registration.register(
    id="nethack_dev-v0", entry_point="silg.envs.nethack:NetHackMultitaskDev"
)
registration.register(
    id="nethack_test-v0", entry_point="silg.envs.nethack:NetHackMultitaskTest"
)

registration.register(
    id="nethack_train_new-v0", entry_point="silg.envs.nethack:NetHackMultitaskNew"
)
registration.register(
    id="nethack_dev_new-v0", entry_point="silg.envs.nethack:NetHackMultitaskNewDev"
)
registration.register(
    id="nethack_test_new-v0", entry_point="silg.envs.nethack:NetHackMultitaskNewTest"
)

registration.register(
    id="nethack_gold-v0", entry_point="silg.envs.nethack:NetHackGold"
)
registration.register(
    id="nethack_scout-v0", entry_point="silg.envs.nethack:NetHackScout"
)
registration.register(
    id="nethack_eat-v0", entry_point="silg.envs.nethack:NetHackEat"
)
registration.register(
    id="nethack_score-v0", entry_point="silg.envs.nethack:NetHackScore"
)


def _step_to_range(delta, num_steps):
    """Range of `num_steps` integers with distance `delta` centered around zero."""
    return delta * torch.arange(-num_steps // 2, num_steps // 2)


class NetHackConvert(SILGEnv):

    def get_text_fields(self):
        return ['goal', 'msg']

    def get_max_actions(self):
        return len(self.action_space)

    def get_observation_space(self):
        return {
            'name': (self.height, self.width, self.max_placement, self.max_grid),
            'name_len': (self.height, self.width, self.max_placement),  # lengths for cell descriptors
            'goal': (self.max_goal, ),  # a vector of word ids that correspond to goal input.
            'goal_len': (1, ),
            'msg': (self.max_msg, ),  # a vector of word ids that correspond to msg input.
            'msg_len': (1, ),
            'valid': (len(self.action_space), ),  # a 1-0 vector that is a mask for valid actions, should be the same length as `self.action_space`
            'rel_pos': (self.height, self.width, 2),  # a matrix of relative distance from each cell to the agent. The 2 entries are the y distance and the x distance, normalized by the height and width of the grid.
            'pos': (2, ),  # agent position (y, x)
        }

    def __init__(self, Base, *args, max_grid=1, max_text=40, time_penalty=-0.02, cache_dir='cache', max_steps=80, renderer='bert_tokenize', **kwargs):
        self.Base = Base
        observation_keys = ["glyphs", "blstats", "tty_chars"]
        if renderer == 'text':
            observation_keys.extend(['tty_colors', 'tty_cursor'])
        base_kwargs = dict(
            savedir=None,
            observation_keys=observation_keys,
            archivefile=None,
            character="mon-hum-neu-mal",
            penalty_step=0,
            penalty_time=0,
        )
        self.Base.__init__(self, **base_kwargs)

        height, width = nethack.DUNGEON_SHAPE
        if height % 2 == 1:
            height += 1
        if width % 2 == 1:
            width += 1

        self.max_grid = max_grid
        self.max_text = max_text
        self.max_goal = 20
        self.max_msg = 20
        self.max_placement = 1
        self.grid_vocab = MAX_GLYPH
        self.action_space = list(range(self.action_space.n))
        super().__init__(height=height, width=width, time_penalty=time_penalty, max_steps=max_steps, renderer=renderer)

    def convert_to_str(self, obs):
        # agent position
        x_offset = torch.Tensor(self.height, self.width).zero_()
        y_offset = torch.Tensor(self.height, self.width).zero_()
        x, y = obs['blstats'][:2]
        for i in range(self.width):
            x_offset[:, i] = i - x
        for i in range(self.height):
            y_offset[i, :] = i - y
        rel_pos = torch.stack([x_offset/self.width, y_offset/self.height], dim=2)
        position = torch.tensor([y, x])

        # text
        msg = bytes(obs['tty_chars'][0].tolist()).decode()
        if '\x00' in msg:
            msg = msg[:msg.index('\x00')]

        # grid
        grid = torch.from_numpy(obs['glyphs'])
        height, width = grid.size()
        if height < self.height:
            grid = torch.cat([grid, torch.zeros(self.height-height, grid.size(1))], dim=0)
        if width < self.width:
            grid = torch.cat([grid, torch.zeros(grid.size(0), self.width-width)], dim=1)
        grid = grid.unsqueeze(2)

        ret = dict(
            name=grid.unsqueeze(-2),
            name_len=torch.ones(self.height, self.width, self.max_placement),
            goal=self.goal,
            msg=msg,
            valid=torch.ones(len(self.action_space)),
            rel_pos=rel_pos,
            pos=position,
        )

        if self.renderer == 'text':
            for k in ['tty_chars', 'tty_colors', 'tty_cursor']:
                ret[k] = obs[k]
        return ret

    def close(self):
        self.Base.close(self)
        tmpdir = self.env._vardir
        if os.path.isdir(tmpdir):
            shutil.rmtree(tmpdir)

    def my_reset(self):
        obs = self.Base.reset(self)
        return self.convert_to_str(obs)

    def my_step(self, *args, **kwargs):
        obs, reward, done, info = self.Base.step(self, *args, **kwargs)
        info = dict(won=True)  # by default individual tasks have no winning condition
        return self.convert_to_str(obs), reward, done, info

    def render_grid(self, obs):
        ns = nethack.tty_render(
            obs["tty_chars"], obs["tty_colors"], obs["tty_cursor"]
        )
        print(ns)

    def parse_user_action(self, inp, obs):
        act = self._actions.index(inp)
        return act

    def get_user_actions(self, obs):
        return {chr(a.value): a.name for a in self._actions}


class NetHackGold(NetHackConvert, NetHackGoldBase):
    goal = 'get more gold'
    Base = NetHackGoldBase

    def __init__(self, *args, **kwargs):
        super().__init__(self.Base, *args, **kwargs)


class NetHackScout(NetHackConvert, NetHackScoutBase):
    goal = 'explore the map'
    Base = NetHackScoutBase

    def __init__(self, *args, **kwargs):
        super().__init__(self.Base, *args, **kwargs)

    def _reward_fn(self, last_observation, action, observation, end_status):
        # this fixes a bug in scout empty cell check, the empty grid id is 2359 and not 0
        del end_status  # Unused
        del action  # Unused

        if not self.env.in_normal_game():
            # Before game started or after it ended stats are zero.
            return 0.0

        reward = 0
        glyphs = observation[self._glyph_index]
        blstats = observation[self._blstats_index]

        dungeon_num = blstats[nethack.NLE_BL_DNUM]
        dungeon_level = blstats[nethack.NLE_BL_DLEVEL]

        key = (dungeon_num, dungeon_level)
        explored = np.sum(glyphs != 2359)
        explored_old = 0
        if key in self.dungeon_explored:
            explored_old = self.dungeon_explored[key]
        reward = explored - explored_old
        self.dungeon_explored[key] = explored
        time_penalty = self._get_time_penalty(last_observation, observation)
        return reward + time_penalty


class NetHackScore(NetHackConvert, NetHackScoreBase):
    goal = 'get high score'
    Base = NetHackScoreBase

    def __init__(self, *args, **kwargs):
        super().__init__(self.Base, *args, **kwargs)


class NetHackEat(NetHackConvert, NetHackEatBase):
    goal = 'eat food'
    Base = NetHackEatBase

    def __init__(self, *args, **kwargs):
        super().__init__(self.Base, *args, **kwargs)


class NetHackMultitask(gym.Env):
    seeds = list(range(0, 1000000))

    supported_envs = [
        NetHackGold,
        # NetHackEat,
        NetHackScout,
        NetHackScore,
    ]
    expert_rewards = [
        [4.62, 4.71, 6.67, 3.68, 6.73],
        # [],
        [170, 50, 85, 67, 80],
        [15, 3.75, 2.75, 2.64, 30],
    ]
    reward_threshold = [np.mean(r) for r in expert_rewards]

    def __init__(self, *args, **kwargs):
        self.envs = [E(*args, **kwargs) for E in self.supported_envs]
        self.current_env = np.random.choice(self.envs)
        self.steps = 0
        self.max_steps = kwargs.get('max_steps', 80)
        self.cumulative_reward = 0

    @property
    def observation_space(self):
        return self.current_env.observation_space

    @property
    def action_space(self):
        return self.current_env.action_space

    def reset(self):
        self.current_env = np.random.choice(self.envs)
        self.steps = 0
        self.cumulative_reward = 0
        seed = np.random.choice(self.seeds)
        self.current_env.seed(seed)
        return self.current_env.reset()

    def step(self, *args, **kwargs):
        obs, reward, done, info = self.current_env.step(*args, **kwargs)
        idx = self.supported_envs.index(self.current_env.__class__)
        reward_threshold = self.reward_threshold[idx]

        info['won'] = False
        self.cumulative_reward += reward

        self.steps += 1
        if self.steps > self.max_steps:
            done = True

        if self.cumulative_reward >= reward_threshold:
            reward = 1
            done = True
            info['won'] = True
        else:
            reward = 0
        if done and self.cumulative_reward < reward_threshold:
            reward = -1
        if reward == 0:
            reward += self.current_env.time_penalty
        return obs, reward, done, info

    def __getattr__(self, name):
        return getattr(self.current_env, name)

    def close(self):
        for env in self.envs:
            env.close()


class NetHackMultitaskDev(NetHackMultitask):

    seeds = list(range(1000000, 2000000))


class NetHackMultitaskTest(NetHackMultitask):

    seeds = list(range(2000000, 3000000))


class NetHackMultitaskNew(NetHackMultitask):

    expert_rewards = [
        [4.38, 4.38, 1.37, 16.38, 2.38, 8.38, 2.38, 0.38],
        # [],
        [428.38, 374.38, 323.38, 301.38, 237.38, 258.38, 293.38, 307.38],
        [102.38, -2.62, 9.38, 10.38, 211.38, 32.38, 85.38, 5.38],
    ]
    reward_threshold = [np.mean(r) for r in expert_rewards]


class NetHackMultitaskNewDev(NetHackMultitaskNew):

    seeds = list(range(1000000, 2000000))


class NetHackMultitaskNewTest(NetHackMultitaskNew):

    seeds = list(range(2000000, 3000000))
