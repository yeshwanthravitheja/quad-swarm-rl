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

    def reset(self, obst_map=None, cell_centers=None, free_space=None, sbc_only_index=-1):
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

        self.update_formation_and_relate_param()

        self.formation_center = np.array((0., 0., 2.))
        self.spawn_points = copy.deepcopy(self.start_point)
        self.goals = copy.deepcopy(self.end_point)

    def generate_end_point(self):
        end_point_list = []
        end_index_list = []
        end_index_2d_list = []

        for qid in range(self.num_agents):
            if self.num_agents >= len(self.free_space):
                pick_list = np.array(self.free_space)
            else:
                pick_list = np.array([row for row in self.free_space if row not in self.start_index_2d_list[qid]])

            end_point = None
            index = None
            index_2d = None
            for idx in range(10):
                end_point, index, index_2d = self.generate_single_point(pick_list=pick_list)
                dist_list = np.linalg.norm(self.start_point - end_point)
                min_dist = np.min(dist_list)
                end_point_pos_bool = abs(end_point[0]) < 4 and abs(end_point[1]) < 4
                if min_dist > 3 and end_point_pos_bool:
                    # print('id: ', idx)
                    # print('end_point: ', end_point)
                    break

            end_point_list.append(end_point)
            end_index_list.append(index)
            end_index_2d_list.append(index_2d)

        return end_point_list, end_index_list, end_index_2d_list

    def generate_single_point(self, pick_list):
        idx = np.random.choice(len(pick_list))
        x, y = pick_list[idx][0], pick_list[idx][1]
        width = self.room_dims[0]
        index = int(x + (width * y))
        pos_x, pos_y = self.cell_centers[index]
        noise = np.random.uniform(low=-0.25, high=0.25, size=2)
        pos_x = pos_x + noise[0]
        pos_y = pos_y + noise[1]
        z_list_start = np.random.uniform(low=1.0, high=3.0)
        point_pos = np.array([pos_x, pos_y, z_list_start])
        return point_pos, index, [x, y]