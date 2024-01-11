import numpy as np

from gym_art.quadrotor_multi.scenarios.base import QuadrotorScenario


class Scenario_o_base(QuadrotorScenario):
    def __init__(self, quads_mode, envs, num_agents, room_dims):
        super().__init__(quads_mode, envs, num_agents, room_dims)
        self.start_point = np.array([0.0, -3.0, 2.0])
        self.end_point = np.array([0.0, 3.0, 2.0])
        self.room_dims = room_dims
        self.quads_mode = quads_mode
        self.free_space = []
        self.approach_goal_metric = 1.0
        self.obstacle_map = None
        self.spawn_points = None
        self.cell_centers = None

    def generate_pos(self):
        half_room_length = self.room_dims[0] / 2
        half_room_width = self.room_dims[1] / 2

        x = np.random.uniform(low=-1.0 * half_room_length + 2.0, high=half_room_length - 2.0)
        y = np.random.uniform(low=-1.0 * half_room_width + 2.0, high=half_room_width - 2.0)
        z = np.random.uniform(low=1.0, high=4.0)

        return np.array([x, y, z])

    def step(self):
        return

    def reset(self, obst_map=None, cell_centers=None, free_space=None):
        self.start_point = self.generate_pos()
        self.end_point = self.generate_pos()
        self.standard_reset(formation_center=self.start_point)

    def generate_pos_obst_map(self, check_surroundings=False):
        idx = np.random.choice(a=len(self.free_space), replace=False)
        x, y = self.free_space[idx][0], self.free_space[idx][1]
        if check_surroundings:
            surroundings_free = self.check_surroundings(x, y)
            while not surroundings_free:
                idx = np.random.choice(a=len(self.free_space), replace=False)
                x, y = self.free_space[idx][0], self.free_space[idx][1]
                surroundings_free = self.check_surroundings(x, y)

        if self.obstacle_map is None:
            width = self.room_dims[0]
        else:
            width = self.obstacle_map.shape[0]
        index = int(x + (width * y))
        pos_x, pos_y = self.cell_centers[index]
        z_list_start = np.random.uniform(low=0.75, high=3.0)
        return np.array([pos_x, pos_y, z_list_start])

    def generate_pos_obst_map_2(self, num_agents):
        generated_points = []
        index_list = []
        index_2d_list = []

        ids = np.random.choice(range(len(self.free_space)), num_agents, replace=True)
        for idx in ids:
            x, y = self.free_space[idx][0], self.free_space[idx][1]
            index_2d_list.append([x, y])
            if self.obstacle_map is None:
                width = self.room_dims[0]
            else:
                width = self.obstacle_map.shape[0]
            index = int(x + (width * y))
            pos_x, pos_y = self.cell_centers[index]
            z_list_start = np.random.uniform(low=1.0, high=3.0)

            index_list.append(index)
            generated_points.append(np.array([pos_x, pos_y, z_list_start]))

        return np.array(generated_points), index_list, index_2d_list

    def check_surroundings(self, row, col):
        length, width = self.obstacle_map.shape[0], self.obstacle_map.shape[1]
        obstacle_map = self.obstacle_map
        # Check if the given position is out of bounds
        if row < 0 or row >= width or col < 0 or col >= length:
            raise ValueError("Invalid position")

        # Check if the surrounding cells are all 0s
        check_pos_x, check_pos_y = [], []
        if row > 0:
            check_pos_x.append(row - 1)
            check_pos_y.append(col)
            if col > 0:
                check_pos_x.append(row - 1)
                check_pos_y.append(col - 1)
            if col < length - 1:
                check_pos_x.append(row - 1)
                check_pos_y.append(col + 1)
        if row < width - 1:
            check_pos_x.append(row + 1)
            check_pos_y.append(col)

        if col > 0:
            check_pos_x.append(row)
            check_pos_y.append(col - 1)
        if col < length - 1:
            check_pos_x.append(row)
            check_pos_y.append(col + 1)
            if row > 0:
                check_pos_x.append(row - 1)
                check_pos_y.append(col + 1)
            if row < length - 1:
                check_pos_x.append(row + 1)
                check_pos_y.append(col + 1)

        check_pos = ([check_pos_x, check_pos_y])
        # Get the values of the adjacent cells
        adjacent_cells = obstacle_map[tuple(check_pos)]

        return np.any(adjacent_cells != 0)
