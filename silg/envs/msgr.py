'''
Wrappers for the messenger environment.
'''
import gym
import torch
import numpy as np
from gym.envs import registration
from .base import SILGEnv
import messenger
from messenger.envs.config import ALL_ENTITIES, STATE_HEIGHT, STATE_WIDTH


# register environments if not registered already
if "msgr_train-all_s1-v0" not in [env.id for env in gym.envs.registry.all()]:
    for stage in range(1, 4):
        for split in ("train-all", "train-sc", "train-mc", "val", "test"):
            registration.register(
                id="msgr_{split}_s{stage}-v0".format(split='dev' if split == 'val' else split, stage=stage),
                entry_point="silg.envs.msgr:MessengerWrapper",
                kwargs=dict(
                    stage=stage,
                    split=split
                )
            )


class MessengerWrapper(SILGEnv):
    '''
    Implements a wrapper for the messenger environment.
    '''
    def get_text_fields(self):
        return ['text', 'goal', 'enemy', 'key']

    def get_max_actions(self):
        return self.env.action_space.n

    def get_observation_space(self):
        return {
            "name": (self.height, self.width, self.max_placement, self.max_grid),
            "name_len": (self.height, self.width, self.max_placement),
            "text": (self.max_text, ),
            "text_len": (1, ),
            "goal": (self.max_goal, ),
            "goal_len": (1, ),
            "enemy": (self.max_enemy, ),
            "enemy_len": (1, ),
            "key": (self.max_key, ),
            "key_len": (1, ),
            "valid": (len(self.action_space), ),
            'pos': (2, ),  # agent position (y, x)
            'rel_pos': (self.height, self.width, 2),  # agent position (y, x)
        }

    def __init__(self, stage: int, split: str, cache_dir="cache", time_penalty: int = 0, max_steps=80, renderer='bert_tokenize', **kwargs):
        if split == "train-all":  # union of multi-comb and single-comb games
            self.env = messenger.envs.TwoEnvWrapper(
                stage=stage,
                split_1="train_mc",
                split_2="train_sc",
                prob_env_1=0.75,
                **kwargs
            )

        if stage == 1:
            if split != "train-all":
                self.env = messenger.envs.StageOne(split=split, **kwargs)
            self.max_steps = 6
            self.max_grid = 2
        elif stage == 2:
            if split != "train-all":
                self.env = messenger.envs.StageTwo(split=split, **kwargs)
            self.max_steps = 64
            self.max_grid = 4
        elif stage == 3:
            if split != "train-all":
                self.env = messenger.envs.StageThree(split=split, **kwargs)
            self.max_steps = 128
            self.max_grid = 6
        else:
            raise Exception("stage must be one of {1,2,3}")

        self.grid_vocab = 20
        self.max_text = 128
        self.max_goal = 40
        self.max_enemy = 40
        self.max_key = 40
        self.max_placement = 1
        self.action_space = list(range(self.env.action_space.n))
        super().__init__(STATE_HEIGHT, STATE_WIDTH, time_penalty=time_penalty, max_steps=max_steps, renderer=renderer)

    def close(self):
        self.env.close()

    def my_reset(self):
        obs, manual = self.env.reset()
        self.enemy_str, self.key_str, self.goal_str = manual
        return self.convert_to_str(obs)

    def convert_to_str(self, obs):
        obs = obs.copy()
        grid = np.concatenate((obs["entities"], obs["avatar"]), axis=-1)
        grid = torch.from_numpy(grid).unsqueeze(-2)
        grid_len = torch.ones(self.height, self.width, self.max_placement) * self.max_grid

        # get the avatar position in the obs
        pos = np.where(obs["avatar"] != 0)
        if not pos[0].tolist():  # agent died
            x = y = 0
        else:
            y, x = pos[0].item(), pos[1].item()

        # compute valid actions at this position
        valid = torch.ones(len(self.action_space))
        # valid[-1] = 0 # in stage 2 this might be important
        if y <= 0:
            valid[0] = 0
        if y >= STATE_HEIGHT - 1:
            valid[1] = 0
        if x <= 0:
            valid[2] = 0
        if x >= STATE_WIDTH - 1:
            valid[3] = 0

        # compute rel_pos
        x_offset = torch.Tensor(self.height, self.width).zero_()
        y_offset = torch.Tensor(self.height, self.width).zero_()
        for i in range(self.width):
            x_offset[:, i] = i - x
        for i in range(self.height):
            y_offset[i, :] = i - y
        rel_pos = torch.stack([x_offset/self.width, y_offset/self.height], dim=2)

        manual = ' '.join([self.enemy_str, self.goal_str, self.key_str])
        ret = dict(
            name=grid,
            name_len=grid_len,
            text=manual,
            goal=self.goal_str,
            key=self.key_str,
            enemy=self.enemy_str,
            valid=valid,
            pos=torch.tensor([y, x]).long(),
            rel_pos=rel_pos,
        )
        return ret

    def my_step(self, action):
        obs, reward, done, info = self.env.step(action)
        info = dict(won=reward == 1)
        return self.convert_to_str(obs), reward, done, info

    def render_grid(self, obs):
        entity_id_to_str = {e.id: e.name for e in ALL_ENTITIES}
        for row in obs['name'][:, :, 0, :].long().tolist():
            for cell in row:
                content = []
                for x in cell:
                    if x != 0:
                        x = entity_id_to_str[x]
                        content.append(x)
                cell = ', '.join(content) if content else '_'
                print('{0:<14}'.format(cell), end='')
            print()

    def my_render_text(self, obs):
        print('DYNAMICS TEXT')
        print(obs[self.TEXT_FIELDS[0]])

        if len(self.TEXT_FIELDS) > 1:
            print('NON DYNAMICS TEXT')
        for t in self.TEXT_FIELDS[1:]:
            v = obs[t]
            t = {
                'key': 'm3',
                'enemy': 'm2',
                'goal': 'm1',
            }[t]
            print('{}: {}'.format(t, v))

    def parse_user_action(self, inp, obs):
        # the order is 'up down left right stay'.split()
        a = chr(inp)
        mapping = {
            'w': 0,
            's': 1,
            'a': 2,
            'd': 3,
            '': 4,
        }
        if a not in mapping:
            raise ValueError('Invalid command {}'.format(a))
        return mapping[a]

    def get_user_actions(self, obs):
        return {'w': 'up', 's': 'down', 'a': 'left', 'd': 'right', '': 'stay'}
