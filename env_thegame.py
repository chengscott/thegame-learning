from math import hypot, pi, sin, asin, atan2
from time import sleep
import subprocess
import multiprocessing

import numpy as np
from gym import spaces
from thegame import HeadlessClient, Ability, Polygon, Bullet, Hero
from server_api import Server


def draw_value(arr, value, center, radius, layer, hero_pos):
  x, y = center
  hx, hy = hero_pos
  x, y, hx, hy = x / 10, y / 10, hx / 10, hy / 10
  x += 80 - hx
  y += 80 - hy
  sx = int(max(0, x - radius))
  sy = int(max(0, y - radius))

  ex = int(min(160, x + radius))
  ey = int(min(160, y + radius))
  for i in range(sx, ex):
    for j in range(sy, ey):
      if (i - x)**2 + (j - y)**2 <= radius**2:
        arr[i, j, layer] = value
  return arr


def to_state(hero, heroes, polygons, bullets):
  state = np.zeros((160, 160, 5))

  state = draw_value(state, hero.health, hero.position, hero.radius, 0,
                     hero.position)
  state = draw_value(state, hero.body_damage, hero.position, hero.radius, 1,
                     hero.position)
  state = draw_value(state, hero.velocity[0], hero.position, hero.radius, 3,
                     hero.position)
  state = draw_value(state, hero.velocity[1], hero.position, hero.radius, 4,
                     hero.position)

  for p in polygons:
    state = draw_value(state, p.health, p.position, p.radius, 0, hero.position)
    state = draw_value(state, p.body_damage, p.position, p.radius, 1,
                       hero.position)
    state = draw_value(state, p.rewarding_experience, p.position, p.radius, 2,
                       hero.position)
    state = draw_value(state, p.velocity[0], p.position, p.radius, 3,
                       hero.position)
    state = draw_value(state, p.velocity[1], p.position, p.radius, 4,
                       hero.position)
  return state


def run_client(pipe_actions, pipe_obv_reward):
  Client.main(pipe_actions, pipe_obv_reward)


class ThegameEnv:
  def __init__(self):
    self.action_space = spaces.Box(
        low=np.array([0.0, 0.0, 0.0]),
        high=np.array([2 * pi, 2 * pi, 8]),
        dtype=np.float32)
    #self.action_space = spaces.Box(shape=(3,), low=-2, high=2)
    self.action_space.dtype = np.dtype('float32')
    self.action_space.n = 3
    self.observation_space = spaces.Box(shape=(160, 160, 5), low=0, high=1000)
    self.observation_space.dtype = np.dtype('int32')
    self.num_envs = 1
    self.server = Server(port=55666)
    self.server.start()
    sleep(0.1)
    self.pipe_obv_reward = multiprocessing.Pipe()
    self.pipe_actions = multiprocessing.Pipe()
    self.client_p = multiprocessing.Process(
        target=run_client,
        args=(
            self.pipe_actions[1],
            self.pipe_obv_reward[0],
        ))
    self.client_p.start()
    sleep(0.1)
    self.counter = 0
    self.prev_score = 0
    self.observation_save = np.zeros((160, 160, 5))

  def reset(self):
    self.server.sync()
    print("reset")
    self.client_p.terminate()
    #self.server.restart()
    sleep(0.5)
    self.client_p = multiprocessing.Process(
        target=run_client,
        args=(
            self.pipe_actions[1],
            self.pipe_obv_reward[0],
        ))
    self.client_p.start()
    sleep(0.5)
    #obv, reward = self.pipe_obv_reward[1].recv()
    self.counter = 0
    return self.observation_save

  def step(self, actions):
    self.pipe_actions[0].send(actions)
    #self.client.set_action(actions)
    self.server.sync()
    obv, reward = self.pipe_obv_reward[1].recv()
    self.observation_save = obv
    done = False
    if self.counter >= 3000:
      done = True
      self.reset()
    self.counter += 1
    info = []
    return obv, np.array([reward]), np.array([done]), info

  def close(self):
    self.client_p.terminate()
    #self.server.shutdown()


class Client(HeadlessClient):
  def init(self):
    self.name = 'atuno'
    self.step = 0
    self.prev_score = 0
    self.reward = 0
    self.acc_dir = 0
    self.shoot_dir = 0
    self.level_up_type = 0
    self.prev_observation = np.zeros((160, 160, 5))
    self.skills = [
        Ability.HealthRegen, Ability.MaxHealth, Ability.MovementSpeed,
        Ability.BulletDamage, Ability.BodyDamage, Ability.Reload,
        Ability.BulletSpeed, Ability.BulletPenetration
    ]
    self.penalty = 0

  def set_action(self, actions):
    acc_dir, shoot_dir, level_up_type = actions
    self.acc_dir = acc_dir
    self.shoot_dir = shoot_dir
    self.level_up_type = level_up_type

  def action(self, hero, heroes, polygons, bullets):
    self.observation = to_state(hero, heroes, polygons, bullets)
    self.reward = hero.score - self.prev_score
    self.prev_score = hero.score

    actions = self.pipe_actions.recv()
    print("recv actions:", actions)
    acc_dir, shoot_dir, level_up_type = actions[0]
    self.penalty = 0
    for act in [acc_dir, shoot_dir, level_up_type]:
      if act < -2:
        self.penalty -= ((-2 - act) + 1)**2
      elif act >= 2:
        self.penalty -= ((act - 2) + 1)**2
    acc_dir = (acc_dir / 2 + 1) * pi
    shoot_dir = (shoot_dir / 2 + 1) * pi
    level_up_type = int((level_up_type + 2) / 4 * 9)
    acc_dir = np.clip(acc_dir, 0, 2 * pi)
    shoot_dir = np.clip(shoot_dir, 0, 2 * pi)
    level_up_type = np.clip(level_up_type, 0, 7)

    self.pipe_obv_reward.send((self.observation, self.reward + self.penalty))

    self.accelerate(acc_dir)
    self.shoot(shoot_dir)
    self.level_up(self.skills[int(level_up_type)])

  def _parse(self):
    self.remote = ':55666'

  def set_pipe(self, pipe_actions, pipe_obv_reward):
    self.pipe_obv_reward = pipe_obv_reward
    self.pipe_actions = pipe_actions

  @classmethod
  def main(cls, pipe_actions, pipe_obv_reward):
    self = cls()
    self._parse()
    self.set_pipe(pipe_actions, pipe_obv_reward)
    self.run()
