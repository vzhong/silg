#
# Must be run with OMP_NUM_THREADS=1
#
'''
For debugging the env using random actions run:
OMP_NUM_THREADS=1 python torchbeast.py --env MiniGrid-MultiRoom-N2-S4-v0 --num_actors 1 --num_threads 1 --random_agent --mode test
'''

import logging
import os
import tqdm
import importlib
import copy
import json

os.environ['OMP_NUM_THREADS'] = '1'

import threading
import time
import timeit
import traceback
import pprint
import typing
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch import multiprocessing as mp

import random
import exp_utils
from silg import envs

from core import environment
from core import prof
from core import vtrace
from expman import Experiment, JSONLogger
from expman.job import SlurmJob


logging.basicConfig(
    format=('[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] '
            '%(message)s'),
    level=0)

Buffers = typing.Dict[str, typing.List[torch.Tensor]]


def get_device(flags):
    if not flags.disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device


def compute_baseline_loss(advantages):
    # Take the mean over batch, sum over time.
    return 0.5 * torch.sum(torch.mean(advantages ** 2, dim=1))


def compute_entropy_loss(logits):
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    entropy_per_timestep = torch.sum(-policy * log_policy, dim=-1)
    return -torch.sum(torch.mean(entropy_per_timestep, dim=1))


def compute_policy_gradient_loss(logits, actions, advantages):
    cross_entropy = F.nll_loss(
        F.log_softmax(torch.flatten(logits, 0, 1), dim=-1),
        target=torch.flatten(actions, 0, 1),
        reduction='none')
    cross_entropy = cross_entropy.view_as(advantages)
    advantages.requires_grad = False
    policy_gradient_loss_per_timestep = cross_entropy * advantages
    return torch.sum(torch.mean(policy_gradient_loss_per_timestep, dim=1))


def act(i: int, free_queue: mp.SimpleQueue, full_queue: mp.SimpleQueue,
        model: torch.nn.Module, buffers: Buffers, initial_agent_state_buffers, flags):
    try:
        logging.info('Actor %i started.', i)
        timings = prof.Timings()  # Keep track of how fast things are.

        Net = importlib.import_module('model.{}'.format(flags.model)).Model
        gym_env = Net.create_env(flags)
        seed = i ^ int.from_bytes(os.urandom(4), byteorder='little')
        gym_env.seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        env = environment.Environment(gym_env)
        env_output = env.initial()

        agent_state = model.initial_state(batch_size=1)
        agent_output, unused_state = model(env_output, agent_state)
        while True:
            index = free_queue.get()
            if index is None:
                break

            # Write old rollout end.
            for key in env_output:
                buffers[key][index][0, ...] = env_output[key]
            for key in agent_output:
                buffers[key][index][0, ...] = agent_output[key]
            for i, tensor in enumerate(agent_state):
                initial_agent_state_buffers[index][i][...] = tensor

            # Do new rollout
            for t in range(flags.unroll_length):
                timings.reset()

                with torch.no_grad():
                    agent_output, agent_state = model(env_output, agent_state)

                timings.time('model')

                action = agent_output['action']

                env_output = env.step(action)

                timings.time('step')

                for key in env_output:
                    buffers[key][index][t + 1, ...] = env_output[key]
                for key in agent_output:
                    buffers[key][index][t + 1, ...] = agent_output[key]

                timings.time('write')
            full_queue.put(index)

        if i == 0:
            logging.info('Actor %i: %s', i, timings.summary())

    except KeyboardInterrupt:
        pass  # Return silently.
    except Exception as e:
        logging.error('Exception in worker process %i', i)
        traceback.print_exc()
        print()
        raise e


def get_batch(free_queue: mp.SimpleQueue,
              full_queue: mp.SimpleQueue,
              buffers: Buffers,
              initial_agent_state_buffers,
              flags,
              timings,
              lock=threading.Lock()) -> typing.Dict[str, torch.Tensor]:
    with lock:
        timings.time('lock')
        indices = [full_queue.get() for _ in range(flags.batch_size)]
        timings.time('dequeue')
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1)
        for key in buffers
    }
    initial_agent_state = (
        torch.cat(ts, dim=1)
        for ts in zip(*[initial_agent_state_buffers[m] for m in indices])
    )
    timings.time('batch')
    for m in indices:
        free_queue.put(m)
    timings.time('enqueue')
    batch = {
        k: t.to(device=get_device(flags), non_blocking=True)
        for k, t in batch.items()
    }
    initial_agent_state = tuple(
        t.to(device=get_device(flags), non_blocking=True) for t in initial_agent_state
    )
    timings.time('device')
    return batch, initial_agent_state


def learn(actor_model,
          model,
          batch,
          initial_agent_state,
          optimizer,
          scheduler,
          flags,
          lock=threading.Lock()):
    """Performs a learning (optimization) step."""
    # logging.info('Learner started on device {}.'.format(initial_agent_state[0].device))
    with lock:
        learner_outputs, unused_state = model(batch, initial_agent_state)

        # Use last baseline value (from the value function) to bootstrap.
        bootstrap_value = learner_outputs['baseline'][-1]

        # At this point, the environment outputs at time step `t` are the inputs
        # that lead to the learner_outputs at time step `t`. After the following
        # shifting, the actions in actor_batch and learner_outputs at time
        # step `t` is what leads to the environment outputs at time step `t`.
        batch = {key: tensor[1:] for key, tensor in batch.items()}
        learner_outputs = {
            key: tensor[:-1]
            for key, tensor in learner_outputs.items()
        }

        rewards = batch['reward']
        if flags.reward_clipping == 'abs_one':
            clipped_rewards = torch.clamp(rewards, -1, 1)
        elif flags.reward_clipping == 'soft_asymmetric':
            squeezed = torch.tanh(rewards / 5.0)
            # Negative rewards are given less weight than positive rewards.
            clipped_rewards = torch.where(rewards < 0, 0.3 * squeezed,
                                          squeezed) * 5.0
        elif flags.reward_clipping == 'none':
            clipped_rewards = rewards

        discounts = (~batch['done']).float() * flags.discounting

        # This could be in C++. In TF, this is actually slower on the GPU.
        vtrace_returns = vtrace.from_logits(
            behavior_policy_logits=batch['policy_logits'],
            target_policy_logits=learner_outputs['policy_logits'],
            actions=batch['action'],
            discounts=discounts,
            rewards=clipped_rewards,
            values=learner_outputs['baseline'],
            bootstrap_value=bootstrap_value)

        # Compute loss as a weighted sum of the baseline loss, the policy
        # gradient loss and an entropy regularization term.
        pg_loss = compute_policy_gradient_loss(learner_outputs['policy_logits'],
                                               batch['action'],
                                               vtrace_returns.pg_advantages)
        baseline_loss = flags.baseline_cost * compute_baseline_loss(
            vtrace_returns.vs - learner_outputs['baseline'])
        entropy_loss = flags.entropy_cost * compute_entropy_loss(
            learner_outputs['policy_logits'])
        aux_loss = learner_outputs['aux_loss'][0]

        total_loss = pg_loss + baseline_loss + entropy_loss + aux_loss

        episode_returns = batch['episode_return'][batch['done']]
        episode_lens = batch['episode_step'][batch['done']]
        won = batch['reward'][batch['done']] > 0.8
        stats = {
            'mean_win_rate': torch.mean(won.float()).item(),
            'mean_episode_len': torch.mean(episode_lens.float()).item(),
            'mean_episode_return': torch.mean(episode_returns).item(),
            'total_loss': total_loss.item(),
            'pg_loss': pg_loss.item(),
            'baseline_loss': baseline_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'aux_loss': aux_loss.item(),
        }

        optimizer.zero_grad()
        model.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 40.0)
        optimizer.step()
        scheduler.step()

        # Interestingly, this doesn't require moving off cuda first?
        actor_model.load_state_dict(model.state_dict())
        return stats


def create_buffers(observation_shapes, num_actions, flags) -> Buffers:
    T = flags.unroll_length
    specs = dict(
        reward=dict(size=(T + 1,), dtype=torch.float32),
        done=dict(size=(T + 1,), dtype=torch.bool),
        episode_return=dict(size=(T + 1,), dtype=torch.float32),
        episode_step=dict(size=(T + 1,), dtype=torch.int32),
        last_action=dict(size=(T + 1,), dtype=torch.int64),
        policy_logits=dict(size=(T + 1, num_actions), dtype=torch.float32),
        baseline=dict(size=(T + 1,), dtype=torch.float32),
        action=dict(size=(T + 1,), dtype=torch.int64),
        aux_loss=dict(size=(T + 1, ), dtype=torch.float32),
    )
    for k, shape in observation_shapes.items():
        if '_emb' in k: # this is for --use_bert where we are retruning bert embeddings
            specs[k] = dict(size=(T + 1, *shape), dtype=torch.float32)
        else:
            specs[k] = dict(size=(T + 1, *shape), dtype=torch.long)
    buffers: Buffers = {key: [] for key in specs}
    for _ in range(flags.num_buffers):
        for key in buffers:
            buffers[key].append(torch.empty(**specs[key]).share_memory_())
    return buffers


class Train(SlurmJob):

    def __init__(self):
        super().__init__()
        self.learner_model = None
        self.optimizer = None
        self.scheduler = None
        self.frames = 0

    def state_dict(self):
        return {
            'model_state_dict': self.learner_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'frames': self.frames,
        }

    def load_state_dict(self, d):
        self.frames = d['frames']
        self.initialize_learner_model()
        self.learner_model.load_state_dict(d['model_state_dict'])
        self.optimizer.load_state_dict(d['optimizer_state_dict'])
        self.scheduler.load_state_dict(d['scheduler_state_dict'])

    def initialize_learner_model(self):
        flags = self.flags
        T = flags.unroll_length
        B = flags.batch_size

        Net = importlib.import_module('model.{}'.format(flags.model)).Model
        env = Net.create_env(flags)
        if self.learner_model is None:
            self.learner_model = Net.make(flags, env).to(device=get_device(flags))

        if self.optimizer is None:
            self.optimizer = torch.optim.RMSprop(
                self.learner_model.parameters(),
                lr=flags.learning_rate,
                momentum=flags.momentum,
                eps=flags.epsilon,
                alpha=flags.alpha)

        if self.scheduler is None:
            def lr_lambda(epoch):
                return 1 - min(epoch * T * B, flags.total_frames) / flags.total_frames

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        return env

    def forward(self, exp_path):
        flags = self.flags
        flags.num_buffers = 2 * flags.num_actors

        self.exp.loggers.append(JSONLogger())
        if flags.wandb:
            from expman.loggers.wandb_logger import WandbLogger
            self.exp.loggers.append(WandbLogger(project=os.path.basename(flags.savedir), name=flags.xpid.replace(':', '_')))
        self.exp.start(delete_existing=False)

        T = flags.unroll_length
        B = flags.batch_size

        env = self.initialize_learner_model()

        Net = importlib.import_module('model.{}'.format(flags.model)).Model
        model = Net.make(flags, env)
        buffers = create_buffers(env.observation_space, len(env.action_space), flags)

        model.share_memory()

        # Add initial RNN state.
        initial_agent_state_buffers = []
        for _ in range(flags.num_buffers):
            state = model.initial_state(batch_size=1)
            for t in state:
                t.share_memory_()
            initial_agent_state_buffers.append(state)

        actor_processes = []
        ctx = mp.get_context('fork')
        free_queue = ctx.SimpleQueue()
        full_queue = ctx.SimpleQueue()

        for i in range(flags.num_actors):
            actor = ctx.Process(
                target=act,
                args=(i, free_queue, full_queue, model, buffers, initial_agent_state_buffers, flags))
            actor.start()
            actor_processes.append(actor)

        learner_model = self.learner_model
        optimizer = self.optimizer
        scheduler = self.scheduler

        stats = {}
        logger = logging.getLogger('logfile')
        stat_keys = [
            'total_loss',
            'mean_episode_return',
            'pg_loss',
            'baseline_loss',
            'entropy_loss',
            'aux_loss',
            'mean_win_rate',
            'mean_episode_len',
        ]
        logger.info('# Step\t%s', '\t'.join(stat_keys))

        def batch_and_learn(i, lock=threading.Lock()):
            """Thread target for the learning process."""
            nonlocal stats
            timings = prof.Timings()
            while self.frames < flags.total_frames:
                timings.reset()
                batch, agent_state = get_batch(free_queue, full_queue, buffers, initial_agent_state_buffers, flags, timings)

                stats = learn(model, learner_model, batch, agent_state, optimizer, scheduler, flags)

                timings.time('learn')
                with lock:
                    to_log = dict(frames=self.frames)
                    to_log.update({k: stats[k] for k in stat_keys})
                    self.exp.log(to_log)
                    self.frames += T * B

            if i == 0:
                logging.info('Batch and learn: %s', timings.summary())

        for m in range(flags.num_buffers):
            free_queue.put(m)

        threads = []
        for i in range(flags.num_threads):
            thread = threading.Thread(
                target=batch_and_learn, name='batch-and-learn-%d' % i, args=(i,))
            thread.start()
            threads.append(thread)

        def val_checkpoint(frames):
            if flags.disable_checkpoint:
                return

            fjob = self.job_checkpoint_path(exp_path)
            logging.info('Testing checkpoint %s', fjob)
            test_flags = copy.deepcopy(flags)
            test_flags.resume = fjob
            test_flags.mode = 'test'
            test_flags.env = flags.val_env

            try:
                test_result = test(test_flags)
                test_result['frames'] = frames
                with self.exp.expdir.joinpath('val.jsonl').open(mode='at') as f:
                    f.write(json.dumps(test_result))
            except Exception as e:
                logging.critical('Error:\n{}'.format(e))
            return test_result

        timer = timeit.default_timer
        try:
            last_checkpoint_time = timer()
            while self.frames < flags.total_frames:
                start_frames = self.frames
                start_time = timer()
                time.sleep(5)

                if timer() - last_checkpoint_time > 15 * 60:  # Save every 15 min.
                    self.checkpoint(exp_path)
                    if flags.val_env:  # if we specified a val environment to run
                        val_checkpoint(self.frames)
                    last_checkpoint_time = timer()

                fps = (self.frames - start_frames) / (timer() - start_time)
                if stats.get('episode_returns', None):
                    mean_return = 'Return per episode: %.1f. ' % stats[
                        'mean_episode_return']
                else:
                    mean_return = ''
                total_loss = stats.get('total_loss', float('inf'))
                logging.info('After %i frames: loss %f @ %.1f fps. %sStats:\n%s',
                             self.frames, total_loss, fps, mean_return,
                             pprint.pformat(stats))
        except KeyboardInterrupt:
            return  # Try joining actors then quit.
        else:
            for thread in threads:
                thread.join()
            logging.info('Learning finished after %d frames.', self.frames)
        finally:
            for _ in range(flags.num_actors):
                free_queue.put(None)
            for actor in actor_processes:
                actor.join(timeout=1)

        self.checkpoint(exp_path)
        self.exp.finish()


def test(flags, verbose=False):
    num_eps = flags.eval_eps  # number of eps to run for
    Net = importlib.import_module('model.{}'.format(flags.model)).Model
    gym_env = Net.create_env(flags)
    env = environment.Environment(gym_env)

    model = Net.make(flags, gym_env)
    model.eval()
    if not flags.random_agent:
        if flags.resume and os.path.isfile(flags.resume):
            checkpoint = torch.load(flags.resume, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            logging.critical('NO FILE TO RESUME FROM! {} does not exist!'.format(flags.resume))

    observation = env.initial()
    returns = []
    won = []
    entropy = []
    ep_len = []
    agent_state = model.initial_state(batch_size=1)
    agent_output, unused_state = model(observation, agent_state)
    start_time = time.time()

    if verbose:
        bar = tqdm.tqdm(total=num_eps)

    while len(won) < num_eps:
        done = False
        steps = 0
        # print('launching episode {}/{}'.format(len(won)+1, num_eps))
        while not done:
            if flags.random_agent:
                action = torch.zeros(1, 1, dtype=torch.int32)
                valid = [i for i, v in enumerate(observation['valid'].flatten().tolist()) if v]
                action[0][0] = random.choice(valid)
                observation = env.step(action)
            else:
                agent_outputs, agent_state = model(observation, agent_state)
                observation = env.step(agent_outputs['action'])
                policy = F.softmax(agent_outputs['policy_logits'], dim=-1)
                log_policy = F.log_softmax(agent_outputs['policy_logits'], dim=-1)
                e = -torch.sum(policy * log_policy, dim=-1)
                entropy.append(e.mean(0).item())

            steps += 1
            done = observation['done'].item()
            if observation['done'].item():
                returns.append(observation['episode_return'].item())
                won.append(observation['reward'][0][0].item() > 0.5)
                ep_len.append(steps)
                agent_state = model.initial_state(batch_size=1)
                if verbose:
                    bar.update(1)
                # logging.info('Episode ended after %d steps. Return: %.1f',
                #              observation['episode_step'].item(),
                #              observation['episode_return'].item())
            if flags.mode == 'test_render':
                sleep_seconds = os.environ.get('DELAY', '0.3')
                print(agent_outputs['action'])
                time.sleep(float(sleep_seconds))

                if observation['done'].item():
                    print('Done: {}'.format('You won!!' if won[-1] else 'You lost!!'))
                    print('Episode steps: {}'.format(observation['episode_step']))
                    print('Episode return: {}'.format(observation['episode_return']))
                    done_seconds = os.environ.get('DONE', None)
                    if done_seconds is None:
                        print('Press Enter to continue')
                        input()
                    else:
                        time.sleep(float(done_seconds))

    env.close()
    fps = steps / (time.time() - start_time)
    logging.info('FPS: %.2f, Average returns over %i episodes: %.2f. Win rate: %.2f. Entropy: %.2f. Len: %.2f', fps, num_eps, sum(returns)/len(returns), sum(won)/len(returns), sum(entropy)/max(1, len(entropy)), sum(ep_len)/len(ep_len))
    return{
        'returns': sum(returns)/len(returns),
        'win_rate': sum(won)/len(returns),
        'lengths': sum(ep_len)/len(ep_len)
    }


def main(flags, **kwargs):
    if flags.mode == 'train':
        exp = Experiment.from_namespace(flags, name_field='xpid', logdir_field='savedir')
        exp.save()
        job = Train()
        job(exp.explog)
    else:
        fout = flags.resume.replace('job.tar', flags.eval_fout)
        assert fout != flags.resume
        out = test(flags, verbose=flags.eval_eps > 200)
        with open(fout, 'wt') as f:
            json.dump(out, f, indent=2)


if __name__ == '__main__':
    parser = exp_utils.get_parser()
    flags = parser.parse_args()
    main(flags)
