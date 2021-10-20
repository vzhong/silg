from pathlib import Path
import functools

import torch
import gym
import os
from gym.envs import registration

import numpy as np
from . import navigator
from ..base import SILGEnv


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

TD_ROOT = os.environ.get('TOUCHDOWN_ROOT', os.path.join(ROOT, 'cache', 'touchdown'))
# register gym environments if not already registered
for split in ("train", "dev", "test"):
    if f"td_res50_{split}-v0" not in [env.id for env in gym.envs.registry.all()]:
        registration.register(
            id=f"td_res50_{split}-v0",
            entry_point="silg.envs.touchdown.gym_wrapper:TDWrapper",
            kwargs=dict(
                features_path=os.path.join(TD_ROOT, 'pca_10.npz'),
                data_json=str(Path(__file__).parent.joinpath(f"data/{split}.json")),
                feat_type='res50',
                path_lengths=os.path.join(TD_ROOT, 'shortest_paths.npz'),
            )
        )

    if f"td_segs_{split}-v0" not in [env.id for env in gym.envs.registry.all()]:
        registration.register(
            id=f"td_segs_{split}-v0",
            entry_point="silg.envs.touchdown.gym_wrapper:TDWrapper",
            kwargs=dict(
                features_path=os.path.join(TD_ROOT, 'maj_ds_a10.npz'),
                data_json=str(Path(__file__).parent.joinpath(f"data/{split}.json")),
                feat_type='segs',
                path_lengths=os.path.join(TD_ROOT, 'shortest_paths.npz'),
            )
        )


def colored(r, g, b, text):
    return "\033[38;2;{};{};{}m{}\033[38;2;255;255;255m".format(r, g, b, text)


CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
           'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
           'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
           'bicycle')

PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
           [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
           [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
           [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
           [0, 80, 100], [0, 0, 230], [119, 11, 32]]


class TDWrapper(SILGEnv):
    '''
    Wrapper to make TDNavigator into a gym env.
    '''

    num_seg_ids = len(CLASSES)

    def get_text_fields(self):
        return ['text']

    def get_max_actions(self):
        return len(self.action_space)

    def get_observation_space(self):
        return {
            "features": (self.height, self.width, self.channels) if self._feat_type != 'segs' else (self.height, self.width),
            "rel_pos": (self.height, self.width, 2),
            "x": (self.max_actions, ),
            "valid": (self.max_actions, ),
            "cur_x": (1, ),
            "text": (self.max_text, ),
            "text_len": (1, ),
            'pos': (2, ),  # agent position (y, x)
        }

    def __init__(self, features_path: str, feat_type: str, path_lengths: str, channels: int = 5, max_actions=5, renderer='bert_tokenize',
                 time_penalty=-0.02, cache_size=10, cache_dir="cache", max_steps=64, test_mode=False, **kwargs):
        '''Parameters:
        features_path (Path):
            Either path to a a folder where the features to load are stored, or a single .npz file with all the features.
        feat_type (str):
            One of "jpg" or "res50". This must match the type of data stored in features_path
        path_lengths (Path):
            npz file which stores a dict of pairwise shortest-path distances between panoramas
        channels (int):
            Number of feature channels to use
        cache_size (int):
            Max size of the cache (in GB). If features_path is a single npz file, this parameter is ignored.
        test_mode (bool):
            if set to true, each call to reset will return a unique example. Will raise exception if more calls to reset()
            are called than there are examples in the current datset split.
        '''
        self.features_path = Path(features_path)  # folder where features are stored
        random_panoid = "__2fASJQNXZ9pH0aCH9L_Q"  # random panoid to test run loader
        total_nodes = 29641  # total number of nodes in TD

        if feat_type == "jpg":
            self.channels = 3  # RGB
        elif feat_type == "segs":
            self.channels = None  # segs have no channel dim. each entry is the class id
        else:
            self.channels = channels
        self._feat_type = feat_type  # one of "jpg" or "res50", or "segs"

        # if features_path is a file, load it into memory
        self.all_features = None
        if self.features_path.is_file():
            print("Loading all features from single file...this might take a while...\n")
            data = np.load(self.features_path, allow_pickle=True)
            self.all_features = data["feats"]
            self.panoid_to_idx = data["panoid_to_idx"].item()
            assert self.all_features.shape[0] == total_nodes

        # run the loader once to get height and width and feat size
        print("Testing feature loading...")
        test_feat = self._feature_loader(random_panoid)
        height, width = test_feat.shape[:2]
        print(f"Features size: {(height, width, self.channels)}\n")
        # estimate the feature size in memory and set cache size if applicable
        self.cache_size = None  # default is no cache
        if cache_size and self.features_path.is_dir():
            print("constructing cache...")
            self.cache_size = cache_size
            feat_size = test_feat.nbytes / 1024 ** 3  # size in GB
            maxsize = int(cache_size // feat_size)
            print(f"{cache_size} GB cache will store {maxsize} elements. ({min(100, int(100 * maxsize / total_nodes))}% coverage)\n")
            cache = functools.lru_cache(maxsize=maxsize)
            self._feature_loader = cache(self._feature_loader)

        # environment settings
        self.max_text = 128
        self.action_space = list(range(max_actions))

        self._text = None
        self._distance = int()  # shortest-path distance to target
        self._cur_xs = torch.Tensor()  # list of current possible x's to choose

        self.nav = navigator.TDNavigator(loader_func=self._feature_loader, **kwargs)
        self.test_mode = test_mode
        self.total_samples = len(self.nav.dataset)
        self.cur_sample = 0

        # path lengths for reward function
        print("loading shortest paths file...this might take a while...\n")
        path_len_data = np.load(path_lengths, allow_pickle=True)  # panoid_to_idx needs to be unpickled
        self.path_lengths = path_len_data["lengths"]
        self.panoid_to_idx = path_len_data["panoid_to_idx"].item()  # map panoid to array index

        super().__init__(height=height, width=width, time_penalty=time_penalty, max_steps=max_steps, renderer=renderer)
        print("finished loading env!")

    def _feature_loader(self, panoid: str):
        if self._feat_type == "jpg":
            from pillow import Image
            im = Image.open(self.features_path.joinpath(f'{panoid}.jpg'))
            return np.asarray(im)

        elif self._feat_type == "segs":
            assert self.all_features is not None
            feats = self.all_features[self.panoid_to_idx[panoid]]
            assert feats.shape == (47, 128)
            return feats

        elif self._feat_type == "res50":
            if self.all_features is not None:
                feats = self.all_features[self.panoid_to_idx[panoid]]
            else:
                data = np.load(str(self.features_path.joinpath(f'{panoid}.npz')))
                feats = data['data']
                assert data['panoid'].item() == panoid
            return feats[:, :, :self.channels]

        else:
            raise Exception('feature type not understood')

    def convert_to_str(self, obs):
        ''' Reformat the obs from TDNavigator. WARNING: We implicitly assume _reformat() is called once
        during reset() and once during step()
        '''
        # convert numpy features to torch. This should be v. fast, since
        # the memory is shared.
        np_feats = obs['features']
        # np_feats.setflags(write=True) # pytorch tensors need to be writeable
        obs['features'] = torch.from_numpy(np_feats)
        obs['cur_x'] = torch.tensor([obs['cur_x']]).long()
        x_len = len(obs['x'])
        obs['valid'] = torch.tensor([1] * x_len + [0] * (self.max_actions - x_len))

        # assert 2 <= x_len <= self.max_actions, f"{self.nav.get_cur_panoid()}"

        # convert obs['x'] to tensor of length self.max_actions padded with 0
        xs = torch.zeros(self.max_actions)
        for i, x in enumerate(sorted(obs['x'])):
            xs[i] = x
        obs['x'] = xs.long()
        self._cur_xs = xs

        position = torch.tensor([self.height // 2, x])

        # add text
        obs.update({
            'text': self._text,
            'pos': position,
        })

        x = obs['cur_x'].item()
        x_offset = torch.Tensor(self.height, self.width).zero_()
        y_offset = torch.Tensor(self.height, self.width).zero_()
        for i in range(self.width):
            x_offset[:, i] = i - x
        rel_pos = torch.stack([x_offset/self.width, y_offset/self.height], dim=2)
        obs['rel_pos'] = rel_pos
        return obs

    def _distance_to_target(self):
        i_1 = self.panoid_to_idx[self.nav.get_cur_panoid()]
        i_2 = self.panoid_to_idx[self.nav.target_panoid]
        return self.path_lengths[i_1, i_2].item()

    def my_reset(self, **kwargs):
        if self.test_mode:
            if self.cur_sample >= self.total_samples:
                raise StopIteration
                # raise Exception(f'Calls to reset() exceed total samples {self.total_samples}')
            kwargs['sample_id'] = self.cur_sample
            self.cur_sample += 1

        obs, manual = self.nav.reset(**kwargs)
        self._distance = self._distance_to_target()
        self._text = manual
        return self.convert_to_str(obs)

    def my_step(self, action):
        x = self._cur_xs[action].item()
        obs = self.nav.step(x)

        # compute reward
        new_dist = self._distance_to_target()
        reward = (self._distance - new_dist) / 10
        self._distance = new_dist

        info = dict(won=False)

        # are we done?
        if self.nav.get_cur_panoid() == self.nav.target_panoid:
            done = True
            reward = 1  # bonus
            info['won'] = True
        elif self.steps_taken >= self.max_steps:
            done = True
            reward = -1  # penalty
        else:
            done = False

        if self.cache_size:
            info['cache_info'] = self._feature_loader.cache_info()

        return self.convert_to_str(obs), reward, done, info

    def close(self):
        if self.cache_size:
            self._feature_loader.cache_clear()
        self.nav.close()

    def render_grid(self, obs):
        keys = []
        for r, g, b in PALETTE:
            keys.append(colored(r, g, b, u"\u2588"))

        for row in obs['features'].tolist():
            chars = [keys[i] for i in row]
            print(''.join(chars))

    def my_render_text(self, obs):
        row = obs['features'][0].tolist()
        position_row = ['_' for _ in row]
        position_row[obs['cur_x'].item()] = 'x'

        action_row = ['_' for _ in row]
        for i, x in enumerate(obs['x'].tolist()[:obs['valid'].sum().item()]):
            action_row[x] = repr(i)
        print(''.join(position_row))
        print(''.join(action_row))
        print('You facing the direction marked by x. Choose the number corresponding to the direction you want to move in.')

        paired = list(zip(CLASSES, PALETTE))
        reps = []
        for c, p in paired:
            s = colored(*p, c)
            reps.append(s)
        print('Colour key:')
        print(', '.join(reps))

        print()
        super().my_render_text(obs)

    def parse_user_action(self, inp, obs):
        # the order is 'up down left right stay'.split()
        a = chr(inp)
        i = int(a)
        keys = self.get_user_actions(obs)
        _ = keys[i]
        return i

    def get_user_actions(self, obs):
        num_valid = obs['valid'].sum().item()
        return {i: 'move in direction' for i in range(num_valid)}
