import torch
import gym
from gym.envs import registration
from .base import SILGEnv
import minihack
import random
from nle.nethack import MAX_GLYPH
from nle import nethack


registration.register(
    id="minihack_train-v0", entry_point="silg.envs.minihack:Minihack",
)


class Minihack(SILGEnv):

    ALL_ENVS = [
        # navigation
        'Room-Monster-5x5-v0',
        'Room-Trap-5x5-v0',
        # exploration, these have different action spaces
        # 'Corridor-R2-v0',
        # 'KeyRoom-S5-v0',
        # planning
        'River-Narrow-v0',
        'HideNSeek-v0',
        # memory
        'Memento-Short-F2-v0',
        'CorridorBattle-v0',
        'MazeWalk-9x9-v0',
    ]

    def get_text_fields(self):
        return ['goal', 'msg']

    def get_max_actions(self):
        return self.all_envs[0].action_space.n

    def get_observation_space(self):
        return {
            'name': (self.height, self.width, self.max_placement, self.max_name),  # a grid of word ids that describe occupants of each cell
            'name_len': (self.height, self.width, self.max_placement),  # lengths for cell descriptors
            'goal': (self.max_goal, ),  # a vector of word ids that correspond to goal input.
            'goal_len': (1, ),
            'msg': (self.max_msg, ),  # a vector of word ids that correspond to msg input.
            'msg_len': (1, ),
            'valid': (self.get_max_actions(), ),  # a 1-0 vector that is a mask for valid actions, should be the same length as `self.action_space`
            'rel_pos': (self.height, self.width, 2),  # agent position (y, x)
            'pos': (2, ),  # agent position (y, x)
        }

    def __init__(self, max_name=1, max_goal=10, max_msg=40, max_steps=100, max_placement=1, time_penalty=-0.01, renderer='bert_tokenize', seed=0):
        observation_keys = minihack.base.MH_DEFAULT_OBS_KEYS
        if renderer == 'text':
            observation_keys += ['tty_chars', 'tty_colors', 'tty_cursor']
        kwargs = dict(
            reward_win=1,
            reward_lose=0,
            penalty_step=0,
            penalty_time=0,
        )
        self.all_envs = [gym.make('MiniHack-' + g, **kwargs) for g in self.ALL_ENVS]
        self.max_name = max_name
        self.max_msg = max_msg
        self.max_goal = max_goal
        self.max_text = 4 + max_name + max_goal
        self.grid_vocab = MAX_GLYPH
        self.max_placement = max_placement

        ospace = self.all_envs[0].observation_space
        self.height, self.width = ospace['glyphs'].shape

        self.rng = random.Random(seed)
        self.current_env = self.rng.choice(self.all_envs)

        super().__init__(self.height, self.width, time_penalty=time_penalty, max_steps=max_steps, renderer=renderer)
        self.max_steps = self.current_env.max_episode_steps

    def seed(self, seed):
        self.rng.seed(seed)

    def get_goal_str(self, env):
        return 'do ' + env.__class__.__name__.replace('MiniHack', '')

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
        msg = bytes(obs['message'].tolist()).decode()
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
            goal=self.get_goal_str(self.current_env),
            msg=msg,
            valid=torch.ones(self.get_max_actions()),
            rel_pos=rel_pos,
            pos=position,
        )

        if self.renderer == 'text':
            for k in ['tty_chars', 'tty_colors', 'tty_cursor']:
                ret[k] = obs[k]
        return ret

    def my_reset(self):
        self.current_env = self.rng.choice(self.all_envs)
        self.max_steps = self.current_env.max_episode_steps
        obs = self.current_env.reset()
        return self.convert_to_str(obs)

    def my_step(self, action):
        obs, reward, done, info = self.current_env.step(action)
        info = dict(won=reward == 1)
        return self.convert_to_str(obs), reward, done, info

    def render_grid(self, obs):
        ns = nethack.tty_render(
            obs["tty_chars"], obs["tty_colors"], obs["tty_cursor"]
        )
        print(ns)

    def parse_user_action(self, inp, obs):
        act = self.current_env._actions.index(inp)
        return act

    def get_user_actions(self, obs):
        return {chr(a.value): a.name for a in self.current_env._actions}
