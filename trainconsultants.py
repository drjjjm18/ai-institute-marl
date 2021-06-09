import numpy as np
import tensorflow as tf
from random import randint
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.specs import array_spec, TensorSpec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories.time_step import TimeStep
from tf_agents.agents import DqnAgent
from typing import Callable
from tf_agents.networks.q_network import QNetwork
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tensorflow.python.framework.tensor_spec import BoundedTensorSpec
from tf_agents.trajectories import trajectory
from statistics import mean


def set_up_grid():
    np_grid = np.zeros((5, 5), dtype=np.int32)
    goal = (randint(0, 4), randint(0, 4))
    while goal == (0, 0) or goal == (4, 4):
        goal = (randint(0, 4), randint(0, 4))
    print('goal= ', goal)
    np_grid[goal] = 3
    np_grid[0][0] = 1
    np_grid[4][4] = 2
    return np_grid.flatten()


def print_grid(np_grid):
    np_grid = np_grid.reshape(5, 5)
    grid_str = '''
    |{}|{}|{}|{}|{}|
    |{}|{}|{}|{}|{}|
    |{}|{}|{}|{}|{}|
    |{}|{}|{}|{}|{}|
    |{}|{}|{}|{}|{}|
    '''.format(*tuple(np_grid.flatten()))
    grid_str = grid_str.replace('1', 'S').replace('0', ' ').replace('2', 'A').replace('3', 'X')
    print(grid_str)


def move_players(np_grid, senior_move, analyst_move):
    np_grid = np_grid.reshape(5, 5)
    senior_pos = np.where(np_grid == 1)
    senior_pos = senior_pos[0][0], senior_pos[1][0]
    analyst_pos = np.where(np_grid == 2)
    analyst_pos = analyst_pos[0][0], analyst_pos[1][0]

    if senior_move == 1:
        new_senior_pos = senior_pos[0] - 1, senior_pos[1]
    elif senior_move == 2:
        new_senior_pos = senior_pos[0], senior_pos[1] + 1
    elif senior_move == 3:
        new_senior_pos = senior_pos[0] + 1, senior_pos[1]
    elif senior_move == 4:
        new_senior_pos = senior_pos[0], senior_pos[1] - 1
    else:
        new_senior_pos = senior_pos

    if any(x < 0 for x in new_senior_pos) or any(x > 4 for x in new_senior_pos):
        new_senior_pos = senior_pos

    if analyst_move == 1:
        new_analyst_pos = analyst_pos[0] - 1, analyst_pos[1]
    elif analyst_move == 2:
        new_analyst_pos = analyst_pos[0], analyst_pos[1] + 1
    elif analyst_move == 3:
        new_analyst_pos = analyst_pos[0] + 1, analyst_pos[1]
    elif analyst_move == 4:
        new_analyst_pos = analyst_pos[0], analyst_pos[1] - 1
    else:
        new_analyst_pos = analyst_pos

    if any(x < 0 for x in new_analyst_pos) or any(x > 4 for x in new_analyst_pos):
        new_analyst_pos = analyst_pos

    if np_grid[new_senior_pos] == np_grid[new_analyst_pos] == 3:
        result = 'draw'

    elif new_analyst_pos == new_senior_pos:
        result = 'continue'

    else:
        if np_grid[new_analyst_pos] == 3:
            result = 'analyst'
            np_grid[analyst_pos] = 0
            np_grid[new_analyst_pos] = 2
        elif np_grid[new_senior_pos] == 3:
            result = 'senior'
            np_grid[senior_pos] = 0
            np_grid[new_senior_pos] = 1
        else:
            result = 'continue'
            np_grid[senior_pos] = 0
            np_grid[new_senior_pos] = 1
            np_grid[analyst_pos] = 0 if np_grid[analyst_pos] != 1 else 1
            np_grid[new_analyst_pos] = 2

    return result, np_grid.flatten()


class GridEnvironment(PyEnvironment):

    def __init__(self):

        super().__init__()
        self._action_spec = array_spec.BoundedArraySpec((2,), np.int32, minimum=1, maximum=5)
        self._observation_spec = array_spec.BoundedArraySpec((25,), np.int32, minimum=0, maximum=4)
        self._episode_ended = False
        self._state = set_up_grid()
        self.steps = 0

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        print(f'===\n- resetting -\n===')
        self._state = set_up_grid()
        self._episode_ended = False
        self.steps = 0
        return ts.restart(np.array(self._state))

    def _step(self, action):
        self.steps += 1
        if self.steps >= 20000:
            return ts.termination(np.array(self._state), -0.5)

        result = move_players(self._state, action[0], action[1])
        self._state = result[1]

        if result[0] == 'draw':
            return ts.transition(np.array(result[1]), 0.5)

        if result[0] == 'continue':
            return ts.transition(np.array(result[1]), 0)

        if result[0] == 'senior':
            self._episode_ended = True
            print('senior wins')
            return ts.termination(np.array(result[1]), 1)

        if result[0] == 'analyst':
            self._episode_ended = True
            print('analyst wins')
            return ts.termination(np.array(result[1]), -1)


class MARLAgent(DqnAgent):

    def __init__(self,
                 env: PyEnvironment,
                 action_spec: array_spec.BoundedArraySpec = None,
                 reward_fn: Callable = lambda time_step: time_step.reward,
                 replay_buffer_max_length: int = 10000,
                 fc_layer_params=(200,),
                 **dqn_kwargs):

        self._env = env
        self._reward_fn = reward_fn
        self._observation_spec = self._env.observation_spec()
        self._action_spec = action_spec or self._env.action_spec()
        self.fc_layer_params = fc_layer_params
        self.q_net = self._build_q_net()
        env_ts_spec = self._env.time_step_spec()

        time_step_spec = TimeStep(
            step_type=env_ts_spec.step_type,
            reward=env_ts_spec.reward,
            discount=env_ts_spec.discount,
            observation=self.q_net.input_tensor_spec
        )

        optimiser = tf.keras.optimizers.Adam()
        super().__init__(time_step_spec,
                         self._action_spec,
                         self.q_net,
                         optimiser,
                         **dqn_kwargs)

        self._policy_state = self.policy.get_initial_state(
            batch_size=self._env.batch_size)

        self._replay_buffer = TFUniformReplayBuffer(
            data_spec=self.collect_data_spec,
            batch_size=self._env.batch_size,
            max_length=replay_buffer_max_length)

    def _build_q_net(self):
        fc_layer_params = self.fc_layer_params

        q_net = QNetwork(
            self._observation_spec,
            self._action_spec,
            fc_layer_params=fc_layer_params)

        q_net.create_variables()
        q_net.summary()

        return q_net

    def reset(self):
        self._policy_state = self.policy.get_initial_state(
            batch_size=self._env.batch_size
        )

    def _augment_time_step(self, time_step: TimeStep) -> TimeStep:
        reward = self._reward_fn(time_step)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        if reward.shape != time_step.reward.shape:
            reward = tf.reshape(reward, time_step.reward.shape)

        return TimeStep(
            step_type=time_step.step_type,
            reward=reward,
            discount=time_step.discount,
            observation=time_step.observation
        )


def analyst_reward_fn(ts: TimeStep):
    return ts.reward * -1 if (ts.reward != 0.5 and ts.reward != -0.5) else ts.reward


grid_env = GridEnvironment()
grid_env = TFPyEnvironment(grid_env)
actions = BoundedTensorSpec.from_spec(array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=4))
senior = MARLAgent(grid_env, action_spec=actions, fc_layer_params=(50, 100, 200))
analyst = MARLAgent(grid_env, action_spec=actions, reward_fn=analyst_reward_fn)

from PIL import Image
import imageio


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def create_gif(grids, file_name: str):
    gif_images = []
    im_dict = {
        0: 'images/black.png',
        1: 'images/SC.png',
        2: 'images/A.png',
        3: 'images/shake.png'
    }
    for grid in grids:
        imgs = [Image.open(im_dict[x]) for x in grid.flatten()]
        new_image = image_grid(imgs, rows=5, cols=5)
        gif_images.append(new_image)

    return imageio.mimsave('gifs/' + file_name + '.gif', gif_images, duration=0.2)


games_per_it = 100
total_its = 1500
total_game_count = 0
iteration_move_avg = []
iteration_sen_loss = []
iteration_ana_loss = []

for iteration in range(total_its):
    print('iteration: ', iteration + 1)
    all_move_counts = []

    for game in range(games_per_it):
        if total_game_count % 1 == 0:
            make_gif = True
            gif_grids = []
        else:
            make_gif = False
        total_game_count += 1

        grid_env.reset()
        senior.reset()
        analyst.reset()
        print(f'game number {game + 1}')
        timeStep = grid_env.current_time_step()
        moves = 0
        while not timeStep.is_last():

            if moves % 500 == 0:
                print('move:', moves)
            if make_gif:
                gif_grids.append(timeStep.observation.numpy().reshape(5, 5))
            moves += 1

            s_action = senior.collect_policy.action(timeStep, senior._policy_state)
            a_action = analyst.collect_policy.action(timeStep, analyst._policy_state)
            combined_action = np.array([s_action[0].numpy(), a_action[0].numpy()]).reshape((1, 2))
            combined_action = s_action.replace(action=combined_action)
            nextStep = grid_env.step(combined_action)
            s_data = trajectory.from_transition(timeStep,
                                                s_action,
                                                nextStep)
            a_data = trajectory.from_transition(analyst._augment_time_step(timeStep),
                                                a_action,
                                                analyst._augment_time_step(nextStep))
            senior._replay_buffer.add_batch(s_data)
            analyst._replay_buffer.add_batch(a_data)
            timeStep = grid_env.current_time_step()

        print(f'game complete, moves = {moves}')
        gif_grids.append(timeStep.observation.numpy().reshape(5, 5))
        if make_gif:
            create_gif(gif_grids, str(total_game_count)) if moves <= 200 else create_gif(gif_grids[-200:],
                                                                                         str(total_game_count))
        print('training...')
        for x in range(moves):
            senior_experience, info = senior._replay_buffer.get_next(sample_batch_size=10, num_steps=2)
            senior_experience = senior.train(senior_experience)
            analyst_experience, info = analyst._replay_buffer.get_next(sample_batch_size=10, num_steps=2)
            analyst_experience = analyst.train(analyst_experience)

        print('training complete')
        all_move_counts.append(moves)

    iteration_move_avg.append(mean(all_move_counts))

sc_checkpoint = tf.train.Checkpoint(q_net=senior.q_net)
sc_checkpoint.save('models/sc_check')
an_checkpoint = tf.train.Checkpoint(q_net=analyst.q_net)
an_checkpoint.save('models/an_check')
