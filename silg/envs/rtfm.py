import torch
from rtfm import featurizer as X
from gym.envs import registration
from .base import SILGEnv
from rtfm import tasks as rtasks


for i in range(1, 5):
    registration.register(
        id="rtfm_train_s{}-v0".format(i), entry_point="silg.envs.rtfm:RTFMS{}".format(i),
    )
    registration.register(
        id="rtfm_test_s{}-v0".format(i), entry_point="silg.envs.rtfm:RTFMS{}Dev".format(i),
    )


class RTFMS1(SILGEnv):

    RTFM_ENV = rtasks.GroupsSimpleStationary

    def get_text_fields(self):
        return ['wiki', 'task', 'inv']

    def get_max_actions(self):
        return len(self.rtfm_env.action_space)

    def get_observation_space(self):
        return {
            'name': (self.height, self.width, self.max_placement, self.max_name),  # a grid of word ids that describe occupants of each cell
            'name_len': (self.height, self.width, self.max_placement),  # lengths for cell descriptors
            'text': (self.max_text, ),  # a vector of word ids that correspond to text input.
            'text_len': (1, ),
            'wiki': (self.max_wiki, ),  # a vector of word ids that correspond to wiki input.
            'wiki_len': (1, ),
            'task': (self.max_task, ),  # a vector of word ids that correspond to task input.
            'task_len': (1, ),
            'inv': (self.max_inv, ),  # a vector of word ids that correspond to inventory input.
            'inv_len': (1, ),
            'valid': (len(self.action_space), ),  # a 1-0 vector that is a mask for valid actions, should be the same length as `self.action_space`
            'rel_pos': (self.height, self.width, 2),  # agent position (y, x)
            'pos': (2, ),  # agent position (y, x)
        }

    def __init__(self, featurizer=X.Concat([X.Text(), X.ValidMoves(), X.Position(), X.RelativePosition()]), room_shape=(6, 6), max_name=8, max_inv=8, max_wiki=80, max_task=40, max_text=80, max_steps=80, partially_observable=False, max_placement=1, shuffle_wiki=False, time_penalty=-0.02, cache_dir='cache', renderer='bert_tokenize'):
        self.rtfm_env = self.RTFM_ENV(
            room_shape=room_shape, featurizer=featurizer, partially_observable=partially_observable, max_placement=max_placement, max_name=max_name, max_inv=max_inv, max_wiki=max_wiki, max_task=max_task, time_penalty=0, shuffle_wiki=shuffle_wiki
        )
        self.max_name = max_name
        self.max_inv = max_inv
        self.max_wiki = max_wiki
        self.max_task = max_task
        self.max_text = max_text
        self.max_placement = max_placement
        self.action_space = self.rtfm_env.action_space
        super().__init__(*room_shape, time_penalty=time_penalty, max_steps=max_steps, renderer=renderer)

    def convert_to_str(self, obs):
        obs = obs.copy()
        for k in ['task', 'wiki', 'inv']:
            toks = self.rtfm_env.vocab.index2word(obs[k][:obs[k+'_len']].tolist())
            toks = toks[:-1]  # remove the RTFM pad token
            obs[k] = ' '.join(toks)
            del obs[k + '_len']

        H, W, K, L = obs['name'].size()
        strs = []
        for i in range(H):
            strs.append([])
            for j in range(W):
                toks = self.rtfm_env.vocab.index2word(obs['name'][i][j][0][:obs['name_len'][i][j][0]-1].tolist())  # remove pad
                strs[i].append(' '.join(toks))
        obs['name'] = strs
        del obs['name_len']
        x, y = obs['position'].tolist()
        obs['pos'] = torch.tensor([y, x])
        del obs['position']
        return obs

    def my_reset(self):
        obs = self.rtfm_env.reset()
        return self.convert_to_str(obs)

    def my_step(self, action):
        obs, reward, done, info = self.rtfm_env.step(action)
        info = dict(won=reward > 0.5)
        return self.convert_to_str(obs), reward, done, info

    def render_grid(self, obs):
        for row in obs['name']:
            for cell in row:
                if cell == 'empty':
                    cell = '_'
                print('{0:<20}'.format(cell), end='')
            print()

    def parse_user_action(self, inp, obs):
        # the order is [E.Stay, E.Up, E.Down, E.Left, E.Right]
        a = chr(inp)
        mapping = {
            '': 0,
            'w': 1,
            's': 2,
            'a': 3,
            'd': 4,
        }
        if a not in mapping:
            raise ValueError('Invalid command {}'.format(a))
        return mapping[a]

    def get_user_actions(self, obs):
        return {'w': 'up', 's': 'down', 'a': 'left', 'd': 'right', '': 'stay'}


class RTFMS1Dev(RTFMS1):
    RTFM_ENV = rtasks.GroupsSimpleStationaryDev


class RTFMS2(RTFMS1):
    RTFM_ENV = rtasks.GroupsSimple


class RTFMS2Dev(RTFMS1):
    RTFM_ENV = rtasks.GroupsSimpleDev


class RTFMS3(RTFMS1):
    RTFM_ENV = rtasks.Groups


class RTFMS3Dev(RTFMS1):
    RTFM_ENV = rtasks.GroupsDev


class RTFMS4(RTFMS1):
    RTFM_ENV = rtasks.GroupsNL


class RTFMS4Dev(RTFMS1):
    RTFM_ENV = rtasks.GroupsNLDev
