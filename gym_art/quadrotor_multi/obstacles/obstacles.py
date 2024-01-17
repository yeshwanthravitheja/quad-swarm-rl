import copy
import numpy as np

from gym_art.quadrotor_multi.obstacles.utils import get_surround_sdfs, get_quad_circle_obst_collision_arr, \
    get_quad_grid_obst_collision_arr, find_zero_chunks, get_cell_centers, get_pos_arr, get_surround_sdfs_grid


class MultiObstacles:
    def __init__(self, obstacle_size=1.0, quad_radius=0.046, room_dims=(10, 10, 10), obst_num=25):
        self.size = obstacle_size
        self.obstacle_radius = obstacle_size / 2.0
        self.quad_radius = quad_radius
        self.pos_arr = []
        self.resolution = 0.1

        self.grid_index = [(i, j) for i in range(10) for j in range(10)]

        self.grid_size = 1.0
        self.grid_num_1d = int(room_dims[0] // self.grid_size)
        self.obstacle_num = obst_num
        self.room_dims = room_dims

        # Fixed parameter
        # shape can be circle or cube
        self.obst_shape = 'grid'
        # For debug or visualization
        self.free_space_list = None

    def reset(self, obs, quads_pos, pos_arr):
        self.pos_arr = copy.deepcopy(np.array(pos_arr))

        quads_sdf_obs = 100 * np.ones((len(quads_pos), 9))
        if self.obst_shape == 'grid':
            quads_sdf_obs = get_surround_sdfs_grid(quad_poses=quads_pos[:, :2], obst_poses=self.pos_arr[:, :2],
                                              quads_sdf_obs=quads_sdf_obs, obst_radius=self.obstacle_radius,
                                              resolution=self.resolution)

        else:
            quads_sdf_obs = get_surround_sdfs(quad_poses=quads_pos[:, :2], obst_poses=self.pos_arr[:, :2],
                                              quads_sdf_obs=quads_sdf_obs, obst_radius=self.obstacle_radius,
                                              resolution=self.resolution)

        obs = np.concatenate((obs, quads_sdf_obs), axis=1)

        return obs

    def step(self, obs, quads_pos):
        quads_sdf_obs = 100 * np.ones((len(quads_pos), 9))
        if self.obst_shape == 'grid':
            quads_sdf_obs = get_surround_sdfs_grid(quad_poses=quads_pos[:, :2], obst_poses=self.pos_arr[:, :2],
                                              quads_sdf_obs=quads_sdf_obs, obst_radius=self.obstacle_radius,
                                              resolution=self.resolution)

        else:
            quads_sdf_obs = get_surround_sdfs(quad_poses=quads_pos[:, :2], obst_poses=self.pos_arr[:, :2],
                                              quads_sdf_obs=quads_sdf_obs, obst_radius=self.obstacle_radius,
                                              resolution=self.resolution)

        obs = np.concatenate((obs, quads_sdf_obs), axis=1)

        return obs

    def collision_detection(self, pos_quads):
        if self.obst_shape == 'circle':
            quad_obst_collisions = get_quad_circle_obst_collision_arr(
                quad_poses=pos_quads[:, :2], obst_poses=self.pos_arr[:, :2], obst_radius=self.obstacle_radius,
                quad_radius=self.quad_radius)
        elif self.obst_shape == 'grid':
            quad_obst_collisions = get_quad_grid_obst_collision_arr(
                quad_poses=pos_quads[:, :2], obst_poses=self.pos_arr[:, :2], obst_radius=self.obstacle_radius,
                quad_radius=self.quad_radius)
        else:
            raise NotImplementedError(f'Unknown obstacle shape: {self.obst_shape}')

        collided_quads_id = np.where(quad_obst_collisions > -1)[0]
        collided_obstacles_id = quad_obst_collisions[collided_quads_id]
        quad_obst_pair = {}
        for i, key in enumerate(collided_quads_id):
            quad_obst_pair[key] = int(collided_obstacles_id[i])

        return collided_quads_id, quad_obst_pair

    def generate_obstacles_grid(self):
        flat_indices = range(len(self.grid_index))
        selected_flat_indices = np.random.choice(a=flat_indices, size=self.obstacle_num, replace=False)
        selected_indices = [self.grid_index[index] for index in selected_flat_indices]
        grid = [[0 for _ in range(self.grid_num_1d)] for _ in range(self.grid_num_1d)]
        for i, j in selected_indices:
            grid[i][j] = 1

        # Get free space list
        # A list consists of at least one chunk of free space.
        # i.e., [[], [], []]
        free_space_list = find_zero_chunks(grid)
        self.free_space_list = np.array(free_space_list)
        valid_free_space_list = []
        max_free_space_size = -1
        for item in free_space_list:
            if len(item) > max_free_space_size:
                valid_free_space_list = item
                max_free_space_size = len(item)

        cell_centers = get_cell_centers(obst_area_length=self.room_dims[0], obst_area_width=self.room_dims[1], grid_size=1.0)

        obst_pos_arr = get_pos_arr(selected_indices=selected_indices, cell_centers=cell_centers,
                                   obst_area_length=self.room_dims[1],
                                   room_height=self.room_dims[2], grid_size=self.grid_size)

        return grid, obst_pos_arr, cell_centers, valid_free_space_list

    def generate_obstacles_grid_u_shape(self):
        # Dimensions of the room and the grids
        room_width, room_height = int(self.room_dims[0]), int(self.room_dims[1])
        grid_width, grid_height = 3, 3

        # Iterate through each grid in the room
        xy_list = []
        for y in range(0, room_height - 2, grid_height):
            for x in range(0, room_width - 2, grid_width):
                xy_list.append(([x+1, y+1]))

        # print(xy_list)

        xy_id_list = [i for i in range(len(xy_list))]
        obst_chunk_num = min(len(xy_id_list), int(self.obstacle_num / 3))
        selected_flat_indices = np.random.choice(a=xy_id_list, size=obst_chunk_num, replace=False)
        selected_indices = [xy_list[index] for index in selected_flat_indices]
        final_pos_arr = []
        grid = [[0 for _ in range(self.grid_num_1d)] for _ in range(self.grid_num_1d)]
        for i, j in selected_indices:
            shift_xy_list, shift_idx = self.generate_chunk()
            for shift_id, shift_xy_item in enumerate(shift_xy_list):
                shift_x, shift_y = shift_xy_item
                real_x, real_y = i + shift_x, j + shift_y

                grid[real_x][real_y] = 1
                final_pos_arr.append([real_x, real_y])

        cell_centers = get_cell_centers(obst_area_length=self.room_dims[0], obst_area_width=self.room_dims[1], grid_size=1.0)

        obst_pos_arr = get_pos_arr(selected_indices=final_pos_arr, cell_centers=cell_centers,
                                   obst_area_length=self.room_dims[1], room_height=self.room_dims[2],
                                   grid_size=self.grid_size)

        free_space_list = find_zero_chunks(grid)
        self.free_space_list = np.array(free_space_list)
        return grid, obst_pos_arr, cell_centers, free_space_list[0]

    def generate_chunk(self):
        idx = np.random.choice(a=range(4))
        if idx == 0:
            shift_xy = [[0, 0], [0, 1], [1, 0]]
        elif idx == 1:
            shift_xy = [[0, 0], [0, 1], [1, 1]]
        elif idx == 2:
            shift_xy = [[0, 0], [1, 0], [1, 1]]
        elif idx == 3:
            shift_xy = [[0, 1], [1, 0], [1, 1]]
        else:
            raise NotImplementedError('Unknown generate_chunk idx: ', idx)
        return shift_xy, idx

    def plot(self, grid):
        plt.figure(figsize=(6, 6))
        plt.imshow(np.array(grid).T, cmap='coolwarm', interpolation='nearest')  # Transposing for correct orientation
        plt.colorbar(label='Grid Value')
        plt.title('Grid with Obstacles (Y-axis Inverted)')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.gca().invert_yaxis()  # Inverting the Y-axis
        plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    obstacle_num = 30
    multi_obstacles = MultiObstacles(room_dims=(11, 11, 5), obst_num=obstacle_num)
    # grid = multi_obstacles.generate_obstacles_grid_u_shape()

    # robots = 8
    # grid_size = 10
    # free_space = np.where(grid == 0)
    # grid_index = [(i, j) for i in range(8) for j in range(8)]
    # flat_indices = range(len(grid_index))
    # selected_flat_indices = np.random.choice(a=flat_indices, size=obstacle_num, replace=False)

    # Example start and end points for robots (can be randomly generated or predefined)
    # start_points = [(np.random.randint(grid_size), np.random.randint(grid_size)) for _ in range(robots)]
    # end_points = [(0, 0) for _ in range(robots)]
    # if multi_obstacles.validate_robot_paths(grid, robots, start_points, end_points):
    #     print("All robot paths are valid.")
    # else:
    #     print("Some robot paths do not intersect with obstacles. Adjusting end points or regenerating grid is needed.")
    grid, obst_pos_arr, cell_centers, _ = multi_obstacles.generate_obstacles_grid_u_shape()
    multi_obstacles.plot(grid)
