import os
import sys
import tty
import gym
import termios
import contextlib
from . import envs
from nle import nethack
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--env', default='silg:rtfm_easy-v0')
parser.add_argument('--time_penalty', default=-0.02, type=float)
parser.add_argument('--max_steps', default=80, type=int)


def clear():
    os.system('cls' if os.name == 'nt' else 'clear')


@contextlib.contextmanager
def no_echo():
    tt = termios.tcgetattr(0)
    try:
        tty.setraw(0)
        yield
    finally:
        termios.tcsetattr(0, termios.TCSAFLUSH, tt)


def get_action():
    with no_echo():
        ch = ord(os.read(0, 1))
    if ch in [nethack.C("c")]:
        print("Received exit code {}. Aborting.".format(ch))
        sys.exit(0)
    return ch


def main():
    args = parser.parse_args()
    env = gym.make(args.env, time_penalty=args.time_penalty, max_steps=args.max_steps, renderer='text')

    obs = env.reset()
    done = False
    reward = cumulative_reward = steps = 0
    historical_cumulative_rewards = []
    while True:
        clear()
        env.render_text(obs)
        print()
        hrep = ', '.join(['{:.2g}'.format(x) for x in historical_cumulative_rewards])
        print('Reward: {:.2g}\tCumulative reward: {:.2g}\tSteps: {}\tDone: {}\tYour historical scores: {}'.format(reward, cumulative_reward, steps, done, hrep))
        print('Type to choose action. Type ? to see action list.')
        raw_action = get_action()
        while True:
            if raw_action == 63:  # intercept '?'
                print('Action list: {}'.format(env.get_user_actions(obs)))
                raw_action = get_action()
            else:
                try:
                    act = env.parse_user_action(raw_action, obs)
                    break
                except ValueError:
                    print('Invalid action {} ({}) - please try again.'.format(chr(raw_action), raw_action))
                    raw_action = get_action()
        obs, reward, done, info = env.step(act)
        cumulative_reward += reward
        steps += 1

        if done:
            obs = env.reset()
            historical_cumulative_rewards.append(cumulative_reward)
            steps = 0
            reward = cumulative_reward = 0
