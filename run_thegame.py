#!/usr/bin/env python3
import sys
from baselines import logger
from baselines.common.cmd_util import make_atari_env, atari_arg_parser
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.ppo2 import ppo2
from baselines.ppo2.policies import CnnPolicy, LstmPolicy, LnLstmPolicy, MlpPolicy
import multiprocessing
import tensorflow as tf
import argparse

from env_thegame import Thegame


def train(num_timesteps, seed, policy, save_interval, load_path, save_path):

  ncpu = multiprocessing.cpu_count()
  if sys.platform == 'darwin': ncpu //= 2
  config = tf.ConfigProto(
      allow_soft_placement=True,
      intra_op_parallelism_threads=ncpu,
      inter_op_parallelism_threads=ncpu)
  config.gpu_options.allow_growth = True  #pylint: disable=E1101
  tf.Session(config=config).__enter__()

  #env = VecFrameStack(make_atari_env(env_id, 8, seed), 4)
  env = Thegame()
  policy = {
      'cnn': CnnPolicy,
      'lstm': LstmPolicy,
      'lnlstm': LnLstmPolicy,
      'mlp': MlpPolicy
  }[policy]
  ppo2.learn(
      policy=policy,
      env=env,
      nsteps=128,
      nminibatches=4,
      lam=0.95,
      gamma=0.99,
      noptepochs=4,
      log_interval=1,
      ent_coef=.01,
      lr=lambda f: f * 2.5e-4,
      cliprange=lambda f: f * 0.1,
      total_timesteps=int(num_timesteps * 1.1),
      save_interval=save_interval,
      load_path=load_path,
      save_path=save_path)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-si",
      "--save_interval",
      help="how many times per save",
      default=0,
      dest="save_interval",
      type=int)
  parser.add_argument(
      "-lp",
      "--load_path",
      help="path to load",
      default=None,
      dest="load_path")
  parser.add_argument(
      "-sp",
      "--save_path",
      help="path to save",
      default='log',
      dest="save_path")
  args = parser.parse_args()
  seed = 0
  num_timesteps = 10e6
  logger.configure()
  train(
      num_timesteps=num_timesteps,
      seed=seed,
      policy='cnn',
      save_interval=args.save_interval,
      load_path=args.load_path,
      save_path=args.save_path)


if __name__ == '__main__':
  main()
