import numpy as np
import copy

from gym_art.quadrotor_multi.scenarios.obstacles.o_base import Scenario_o_base


class Scenario_o_static_same_goal(Scenario_o_base):
    def __init__(self, quads_mode, envs, num_agents, room_dims):
        super().__init__(quads_mode, envs, num_agents, room_dims)
        # self variables
        self.approach_goal_metric = 1.0
        self.free_space = None
        self.cell_centers = None
        self.index_list = None
        self.start_index_list = self.start_index_2d_list = None
        self.end_index_list = self.end_index_2d_list = None

    def step(self):
        return

    def reset(self, obst_map=None, cell_centers=None, free_space=None):
        if obst_map is not None:
            self.obstacle_map = obst_map
            obst_map_locs = np.where(self.obstacle_map == 0)
            self.free_space = list(zip(*obst_map_locs))
        elif free_space is not None:
            self.free_space = free_space
        else:
            raise NotImplementedError('obst_map is None and free_space is None')

        self.cell_centers = cell_centers
        self.start_point, self.start_index_list, self.start_index_2d_list = self.generate_pos_obst_map_2(num_agents=self.num_agents)
        self.end_point, self.end_index_list, self.end_index_2d_list = self.generate_end_point()

        # Reset formation and related parameters
        self.update_formation_and_relate_param()
        # Reassign goals
        self.spawn_points = copy.deepcopy(self.start_point)
        self.goals = np.array([self.end_point for _ in range(self.num_agents)])

    def generate_end_point(self):
        end_point = index = index_2d = None
        if self.num_agents >= len(self.free_space):
            end_point, index, index_2d = self.generate_single_point()
        else:
            for _ in range(10):
                end_point, index, index_2d = self.generate_single_point()
                if index not in self.start_index_list:
                    break

        return end_point, index, index_2d

    def generate_single_point(self):
        idx = np.random.choice(len(self.free_space))
        x, y = self.free_space[idx][0], self.free_space[idx][1]
        width = self.room_dims[0]
        index = int(x + (width * y))
        pos_x, pos_y = self.cell_centers[index]
        z_list_start = np.random.uniform(low=1.0, high=3.0)
        point_pos = np.array([pos_x, pos_y, z_list_start])
        return point_pos, index, [x, y]
