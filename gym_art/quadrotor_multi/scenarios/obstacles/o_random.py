import numpy as np
import copy

from gym_art.quadrotor_multi.scenarios.obstacles.o_base import Scenario_o_base


class Scenario_o_random(Scenario_o_base):
    def __init__(self, quads_mode, envs, num_agents, room_dims):
        super().__init__(quads_mode, envs, num_agents, room_dims)
        self.approach_goal_metric = 0.5
        self.start_index_list = self.start_index_2d_list = None
        self.end_index_list = self.end_index_2d_list = None

    def update_formation_size(self, new_formation_size):
        pass

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
        self.end_point, self.end_index_list, self.end_index_2d_list = self.generate_pos_obst_map_2(num_agents=self.num_agents)

        self.update_formation_and_relate_param()

        self.formation_center = np.array((0., 0., 2.))
        self.spawn_points = copy.deepcopy(self.start_point)
        self.goals = copy.deepcopy(self.end_point)
