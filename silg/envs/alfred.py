import os
import torch
import string
import re
import yaml
from gym.envs import registration
from .base import SILGEnv
import alfworld.agents.environment as environment


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
os.environ['ALFWORLD_DATA'] = os.path.join(ROOT, 'cache', 'alfworld')


ITEM_RE = re.compile(r'a (\w+ \d+)')


registration.register(
    id="alfworld_train-v0", entry_point="silg.envs.alfred:TWAlfred"
)
registration.register(
    id="alfworld_test_id-v0", entry_point="silg.envs.alfred:TWAlfredTestID"
)
registration.register(
    id="alfworld_test_od-v0", entry_point="silg.envs.alfred:TWAlfredTestOD"
)
registration.register(
    id="alfworld_test_thor_id-v0", entry_point="silg.envs.alfred:TWAlfredTestThorID"
)
registration.register(
    id="alfworld_test_thor_od-v0", entry_point="silg.envs.alfred:TWAlfredTestThorOD"
)
registration.register(
    id="alfwolrd_test_thor_astar_id-v0", entry_point="silg.envs.alfred:TWAlfredTestThorAStarID"
)
registration.register(
    id="alfworld_test_thor_astar_od-v0", entry_point="silg.envs.alfred:TWAlfredTestThorAStarOD"
)


class TWAlfred(SILGEnv):
    SPLIT = 'train'
    CONFIG_FILE = 'alfred_config_oracle.yaml'
    ENV_TYPE = 'AlfredTWEnv'

    def get_text_fields(self):
        return ['text', 'feedback', 'goal', 'history']

    def get_max_actions(self):
        return len(self.action_space)

    def get_observation_space(self):
        return {
            'name': (self.height, self.width, self.max_placement, self.max_name),  # a grid of word ids that describe occupants of each cell
            'name_len': (self.height, self.width, self.max_placement),  # lengths for cell descriptors
            'text': (self.max_text, ),  # a vector of word ids that correspond to text input.
            'text_len': (1, ),
            'goal': (self.max_goal, ),  # a vector of word ids that correspond to text input.
            'goal_len': (1, ),
            'feedback': (self.max_feedback, ),  # a vector of word ids that correspond to text input.
            'feedback_len': (1, ),
            'history': (self.max_history, ),  # a vector of word ids that correspond to text input.
            'history_len': (1, ),
            'command': (self.max_actions, self.max_action_tok),  # commands
            'command_len': (self.max_actions, ),  # command lengths
            'valid': (self.max_actions, ),  # valid commands
            'rel_pos': (self.height, self.width, 2),  # a matrix of relative distance from each cell to the agent. The 2 entries are the y distance and the x distance, normalized by the height and width of the grid.
            'pos': (2, ),  # agent position (y, x)
        }

    def __init__(self, max_objects=40, max_name=6, max_action=60, max_action_tok=10, max_text=40, time_penalty=0, max_steps=50, cache_dir='cache', reward_explore=1e-4, renderer='bert_tokenize'):
        # load config
        config_file = os.path.join(ROOT, 'cache', self.CONFIG_FILE)
        assert os.path.exists(config_file), "Invalid config file {}".format(config_file)
        with open(config_file) as reader:
            config = yaml.safe_load(reader)
        # env_type = config['env']['type']  # 'AlfredTWEnv' or 'AlfredThorEnv' or 'AlfredHybrid'
        env_type = self.ENV_TYPE

        # setup environment
        env = getattr(environment, env_type)(config, train_eval=self.SPLIT)
        self.alfred_env = env.init_env(batch_size=1)

        # self.max_episode_step = config["rl"]["training"]["max_nb_steps_per_episode"]
        self.max_objects = max_objects
        self.height, self.width = max_objects, 2
        self.max_name = max_name
        self.max_text = max_text
        self.max_goal = 30
        self.max_feedback = 10
        self.max_history = 20
        self.max_placement = 1
        self.max_action_tok = max_action_tok
        self.reward_explore = reward_explore

        self.action_space = list(range(max_action))  # this is a dummy placeholder because the number of actions is dynamic
        self.last_obs = None
        self.last_admissible_commands = None
        self.goal = None
        self.initial_scene = None
        self.history = []
        self.seen_obs = set()

        super().__init__(height=max_objects, width=2, time_penalty=time_penalty, max_steps=max_steps, renderer=renderer)

    def my_reset(self):
        batch_obs, batch_info = self.alfred_env.reset()
        terms = [t.strip() for t in batch_obs[0].split('\n') if t.strip()]
        self.goal = terms[-1].lower().replace('your task is to:', '').strip()

        self.last_obs = '\n'.join(terms[1:-1])
        feedback, self.initial_scene = self.split_feedback(self.last_obs)
        obs = dict(
            feedback=feedback,
            items=[],
            commands=batch_info['admissible_commands'][0][:self.max_actions],
        )
        if 'look' in obs['commands']:
            obs['commands'].remove('look')
        self.last_admissible_commands = obs['commands']
        self.history.clear()
        self.seen_obs.add(self.last_obs)
        # unused keys: won, extra.gamefile, expert_type
        return self.convert_to_str(obs)

    def split_feedback(self, feedback):
        items = ITEM_RE.findall(feedback)
        feedback = feedback.lower()
        if 'you see' in feedback:
            feedback = feedback[:feedback.index('you see')]
        return feedback, items

    def my_step(self, action):
        # the real env is batched w/ batch size 1.
        real_action = self.last_admissible_commands[action]
        # print(self.goal)
        # print(self.last_admissible_commands)
        # print(real_action)
        batch_obs, batch_reward, batch_done, batch_info = self.alfred_env.step([real_action])
        self.last_obs = batch_obs[0]
        # print(batch_reward[0], batch_win[0])
        # print(batch_obs[0])
        feedback, items = self.split_feedback(self.last_obs)
        obs = dict(
            items=items,
            feedback=feedback,
            commands=batch_info['admissible_commands'][0][:self.max_actions],
        )
        done = batch_done[0]
        if 'look' in obs['commands']:
            obs['commands'].remove('look')
        self.last_admissible_commands = obs['commands']
        # reward = batch_reward[0]  -- NOTE: this is broken
        reward = 1 if batch_info['won'][0] else 0
        info = {k: v[0] for k, v in batch_info.items()}

        reward += self.reward_explore if self.last_obs not in self.seen_obs else 0
        converted = self.convert_to_str(obs)

        if not feedback.lower().strip().startswith('nothing happens'):
            self.history.insert(0, feedback)
        if len(self.history) > 3:
            self.history.pop()
        self.seen_obs.add(self.last_obs)

        return converted, reward, done, info

    def convert_to_str(self, obs):
        # build grid of observations
        cells = [['' for _ in range(self.width)] for _ in range(self.height)]

        for i, item in enumerate(self.initial_scene[:self.max_objects]):
            cells[i][0] = item
        for i, item in enumerate(obs['items'][:self.max_objects]):
            cells[i][1] = item

        goal = self.goal
        feedback = obs['feedback']
        history = []
        for h in self.history:
            history.append(h)
        history = '; '.join(history)

        text = '{} feedback {} history {}'.format(goal, feedback, history)

        # err = self.tokenizer.convert_ids_to_tokens(text)
        # raise Exception(repr(err))

        command = []
        command_len = []
        valid = []
        for c in obs['commands']:
            t = self.tokenizer.encode(c)[:self.max_action_tok]
            command_len.append(len(t))
            t += [self.tokenizer.pad_token_id] * (self.max_action_tok - len(t))
            command.append(t)
            valid.append(1)
        command_pad = [self.tokenizer.pad_token_id] * self.max_action_tok
        command = command[:self.max_actions] + (self.max_actions - len(command)) * [command_pad]
        command_len = command_len[:self.max_actions] + (self.max_actions - len(command_len)) * [1]

        valid = valid[:self.max_actions] + (self.max_actions - len(valid)) * [0]

        # concatenate text inputs
        ret = dict(
            name=cells,
            text=text,
            goal=goal,
            feedback=feedback,
            history=history,
            command=torch.tensor(command),
            command_len=torch.tensor(command_len),
            valid=torch.tensor(valid),
            rel_pos=torch.zeros(self.height, self.width, 2),
            pos=torch.zeros(2),
        )
        return ret

    def render_grid(self, obs):
        initial_items = []
        current_items = []
        for i, c in obs['name']:
            if i:
                initial_items.append(i)
            if c:
                current_items.append(c)
        print('Initial items in scene: {}'.format(', '.join(initial_items)))
        print('Current items in scene: {}'.format(', '.join(current_items)))
        print()

    def my_render_text(self, obs):
        super().my_render_text(obs)
        print('Admissible commands')
        keys = list(self.get_user_actions(obs).items())
        per_row = 2
        for i in range(0, len(keys), per_row):
            batch = keys[i:i+per_row]
            for k, v in batch:
                s = '{}: {}'.format(k, v)
                print('{0:<40}'.format(s), end='')
            print()
        print()

    def parse_user_action(self, inp, obs):
        # the order is 'up down left right stay'.split()
        a = chr(inp)
        keys = self.get_user_actions(obs)
        cmd = keys[a]
        return self.last_admissible_commands.index(cmd)

    def get_user_actions(self, obs):
        keys = string.ascii_lowercase + string.ascii_uppercase + string.digits
        assert self.max_actions < len(keys), 'not enough keys! need {} but have {}'.format(self.max_actions, len(keys))
        keys = keys[:self.max_actions]
        return dict(zip(keys, self.last_admissible_commands))


class TWAlfredTestID(TWAlfred):
    SPLIT = 'eval_in_distribution'


class TWAlfredTestOD(TWAlfred):
    SPLIT = 'eval_out_of_distribution'


class TWAlfredTestThorID(TWAlfredTestID):
    ENV_TYPE = 'AlfredThorEnv'


class TWAlfredTestThorOD(TWAlfredTestOD):
    ENV_TYPE = 'AlfredThorEnv'


class TWAlfredTestThorAStarID(TWAlfredTestID):
    ENV_TYPE = 'AlfredThorEnv'
    CONFIG_FILE = 'alfred_config_astar.yaml'


class TWAlfredTestThorAStarOD(TWAlfredTestOD):
    ENV_TYPE = 'AlfredThorEnv'
    CONFIG_FILE = 'alfred_config_astar.yaml'
