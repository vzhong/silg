import gym
import torch
import pathlib
from transformers import BertTokenizerFast
from vocab import Vocab


TOKENIZER_CACHE = pathlib.Path(__file__).parent.parent.parent.joinpath('cache').joinpath('tokenizer')


class SILGEnv(gym.Env):

    def get_text_fields(self):
        raise NotImplementedError()

    def get_max_actions(self):
        raise NotImplementedError()

    def my_reset(self):
        raise NotImplementedError()

    def my_step(self, action):
        raise NotImplementedError()

    def __init__(self, height, width, time_penalty, max_steps, renderer='bert_tokenize'):
        self.height = height
        self.width = width
        self.time_penalty = time_penalty
        self.max_steps = max_steps

        self.TEXT_FIELDS = self.get_text_fields()
        self.max_actions = self.get_max_actions()
        self.observation_space = self.get_observation_space()

        self.renderer = renderer
        self.tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER_CACHE)
        words = [None] * len(self.tokenizer)
        for word, index in self.tokenizer.get_vocab().items():
            words[index] = word
        self.vocab = Vocab(words)

    def parse_user_action(self, inp, obs):
        raise NotImplementedError()

    def get_user_actions(self, obs):
        raise NotImplementedError()

    def render_grid(self, obs):
        for row in obs['name']:
            for cell in row:
                print('{0:<20}'.format(cell), end='')
            print()

    def my_render_text(self, obs):
        print('JOINT TEXT')
        print(obs[self.TEXT_FIELDS[0]])

        if len(self.TEXT_FIELDS) > 1:
            print('FIELD TEXT')
        for t in self.TEXT_FIELDS[1:]:
            print('{}: {}'.format(t, obs[t]))

    def render_text(self, obs):
        self.render_grid(obs)
        print()
        self.my_render_text(obs)

    def reset(self):
        obs = self.my_reset()
        self.steps_taken = 0
        return self.reformat(obs)

    def step(self, action):
        obs, reward, done, info = self.my_step(action)
        self.steps_taken += 1
        if not done and self.steps_taken > self.max_steps:
            done = True

        if done and not info['won']:
            reward = -1

        if not done:
            reward = reward + self.time_penalty
        return self.reformat(obs), reward, done, info

    def reformat(self, obs):
        ret = obs.copy()
        if self.renderer == 'bert_tokenize':
            for k in self.TEXT_FIELDS:
                max_length = self.observation_space[k][-1]
                encoding = self.tokenizer(text=obs[k], truncation=True, max_length=max_length, padding='max_length', return_tensors='pt')
                ret[k+'_len'] = encoding['attention_mask'].sum(-1)
                ret[k] = encoding['input_ids'][0]

        if hasattr(self, 'grid_vocab'):
            # this env provides non word entity IDs
            assert 'name' in obs
            ret['name'] = obs['name']
            ret['name_len'] = obs['name_len']
        elif 'features' in obs:
            # this env provides pre-extracted non-embeddings features
            pass
        elif self.renderer == 'bert_tokenize':
            # must tokenize name field
            name = []
            lengths = []
            for row in obs['name']:
                encoding = self.tokenizer(row, truncation=True, max_length=self.max_name, padding='max_length', return_tensors='pt')
                name.append(encoding['input_ids'])
                lengths.append(encoding['attention_mask'].sum(-1))
            ret['name'] = torch.stack(name).unsqueeze(-2)
            ret['name_len'] = torch.stack(lengths).unsqueeze(-1)
        return ret
