import gym
import numpy as np
from gym import spaces
N_DISCRETE_ACTIONS = 5
N_OBSERVATION_SCALE = 2
WALL = -5
BLANK_SPACE = 0
REWARD_POINT = 5
AGENT_POINT = 1


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    # current 1: wall 0: blank space 2: reward point 3: agent point
    # in this environment agents can overlap together
    # in this environment agents and reward points can overlap together
    # don't know whether give each agent a unique label
    # if all the reward points are eaten, then reset the reward points

    def __init__(
            self,
            n_agents: int,
    ):
        super(CustomEnv, self).__init__()
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.observation_space = spaces.Box(low=-10, high=10, shape=(21, 21), dtype=np.uint8)
        self.agent_num = n_agents
        # Here initial maze with tini size
        # TODO: add more suitable environments

        self.env_matrix = WALL * np.ones([21, 21])
        self.init_env_matrix()
        self.time_limit = 50
        self.time_step = 0

    def init_env_matrix(self, mode='tini'):
        if mode == 'tini':
            assert (self.env_matrix.shape[0] == 21)

            # first initial the environment
            self.env_matrix[2:7, 9:12] = BLANK_SPACE
            self.env_matrix[7:10, 10] = BLANK_SPACE
            self.env_matrix[10:13, 9:12] = BLANK_SPACE
            self.env_matrix[11, 7:9] = BLANK_SPACE
            self.env_matrix[11, 12:14] = BLANK_SPACE
            self.env_matrix[10:13, 14:19] = BLANK_SPACE
            self.env_matrix[10:13, 2:7] = BLANK_SPACE
            self.env_matrix[13, 10] = BLANK_SPACE
            self.env_matrix[14:19, 9:12] = BLANK_SPACE
            self.agent_position = []

            # second initial the environment
            self.reward_point_set_current = []
            self.reward_point_set_current += self.init_reward_point(4, 3, 3, 2, 9)
            self.reward_point_set_current += self.init_reward_point(4, 3, 3, 15, 9)
            self.reward_point_set_current += self.init_reward_point(3, 4, 3, 10, 15)
            self.reward_point_set_current += self.init_reward_point(3, 4, 3, 10, 2)

            # third initial agents
            self.init_agent_point(3, 3, 10, 9)

    def init_reward_point(self, length, wide, num, init_length, init_wide):
        assert (length != 0 and wide != 0 and num != 0)
        point_set = []
        while len(point_set) != num:
            point = [np.random.randint(length) + init_length, np.random.randint(wide) + init_wide]
            if point not in point_set:
                point_set.append(point)

        for item in point_set:
            self.env_matrix[item[0], item[1]] = REWARD_POINT

        return point_set

    def init_agent_point(self, length, wide, init_length, init_wide):
        assert (length != 0 and wide != 0)
        point_set = []
        while len(point_set) != self.agent_num:
            point = [np.random.randint(length) + init_length, np.random.randint(wide) + init_wide]
            if point not in point_set:
                point_set.append(point)

        for item in point_set:
            self.agent_position.append(np.array(item))
            # self.env_matrix[item[0], item[1]] = 3
            # here agents' positions are not counted in the environment matrix
            # when calculate each agent's observation
            # 1. give matrix information in raleted scale
            # 2. overlap agents' positions if they are in current agent's observation scale
            # the weight of agents is larger then it of environment information such as reward points

    def step(self, action):
        # Execute one time step within the environment
        # Here are five actions
        # 0: up 1: down 2: right 3: left 4: eat

        assert (len(action) == self.agent_num)
        eaten_reward_points = []
        eaten_reward_points_array = []
        return_reward = -0.1 * np.ones(4)
        self.time_step += 1

        for id, item in enumerate(action):
            # first analyse

            if item == 0:
                delta_xy = np.array([-1, 0])
                ifeat = False

            elif item == 1:
                delta_xy = np.array([1, 0])
                ifeat = False

            elif item == 2:
                delta_xy = np.array([0, 1])
                ifeat = False

            elif item == 3:
                delta_xy = np.array([0, -1])
                ifeat = False

            elif item == 4:
                delta_xy = np.array([0, 0])
                ifeat = True

            else:
                raise (print(f'wrong action label: agent{id} action is {item}'))

            next_state = self.agent_position[id] + delta_xy
            if self.env_matrix[next_state[0], next_state[1]] != 1:
                # check if next state is in the wall
                self.agent_position[id] = next_state

            if ifeat:
                if self.env_matrix[self.agent_position[id][0], self.agent_position[id][1]] == 2:
                    # this agent want to eat the reward point
                    eaten_reward_points.append([id, self.agent_position[id]])
                    eaten_reward_points_array.append(self.agent_position[id])

        eaten_reward_points_array = np.array(eaten_reward_points_array)

        # exclude more then one agents eat the same reward point
        for item in eaten_reward_points:
            index_0 = np.where(eaten_reward_points_array[:, 0] == item[1][0])[0]
            index_1 = np.where(eaten_reward_points_array[index_0, 1] == item[1][1])[0]
            return_reward[item[0]] = 1 / len(index_1)
            # more agents choose to eat the same reward point
            # then the reward get by each agent shrinks proportionally

        # remove the reward point
        for item in eaten_reward_points:
            self.env_matrix[item[1][0], item[1][1]] = 0
            if item[1].tolist() in self.reward_point_set_current:
                self.reward_point_set_current.remove(item[1].tolist())

        done = False
        if self.time_step >= self.time_limit:
            # done only if time steps exceed limitation
            done = True

        if len(self.reward_point_set_current) == 0:
            # all current reward points are eaten
            # regenerate reward points
            self.reward_point_set_current += self.init_reward_point(4, 3, 3, 2, 9)
            self.reward_point_set_current += self.init_reward_point(4, 3, 3, 15, 9)
            self.reward_point_set_current += self.init_reward_point(3, 4, 3, 10, 15)
            self.reward_point_set_current += self.init_reward_point(3, 4, 3, 10, 2)

        next_obs = [self.get_local_observation(i) for i in range(self.agent_num)]
        info = {}

        return next_obs, return_reward, done, info

    def get_local_observation(self, id):
        current_position = self.agent_position[id]
        length = N_OBSERVATION_SCALE * 2 + 1
        min_index_0 = current_position[0] - N_OBSERVATION_SCALE
        min_index_1 = current_position[1] - N_OBSERVATION_SCALE

        obs = self.env_matrix[min_index_0:min_index_0 + length, min_index_1:min_index_1 + length]
        for i in range(self.agent_num):
            if i != id:
                other_agent_position = self.agent_position[i]
                if other_agent_position[0] >= min_index_0 and other_agent_position[0] < min_index_0 + length:
                    if other_agent_position[1] >= min_index_1 and other_agent_position[1] < min_index_1 + length:
                        obs[other_agent_position[0] - min_index_0][other_agent_position[1] - min_index_1] = 3

        return obs

    def reset(self):
        # Reset the state of the environment to an initial state
        self.init_env_matrix()
        self.time_step = 0

        obs = [self.get_local_observation(i) for i in range(self.agent_num)]
        return obs

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        pass

    def seed(self, seed=None):
        pass

    def close(self):
        pass
