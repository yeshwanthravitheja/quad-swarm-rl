import numpy as np
from numba import njit
import matplotlib.pyplot as plt


@njit
def get_surround_sdfs(quad_poses, obst_poses, quads_sdf_obs, obst_radius, resolution=0.1):
    # Shape of quads_sdf_obs: (quad_num, 9)

    sdf_map = np.array([-1., -1., -1., 0., 0., 0., 1., 1., 1.])
    sdf_map *= resolution

    for i, q_pos in enumerate(quad_poses):
        q_pos_x, q_pos_y = q_pos[0], q_pos[1]

        for g_i, g_x in enumerate([q_pos_x - resolution, q_pos_x, q_pos_x + resolution]):
            for g_j, g_y in enumerate([q_pos_y - resolution, q_pos_y, q_pos_y + resolution]):
                grid_pos = np.array([g_x, g_y])

                min_dist = 100.0
                for o_pos in obst_poses:
                    dist = np.linalg.norm(grid_pos - o_pos)
                    if dist < min_dist:
                        min_dist = dist

                g_id = g_i * 3 + g_j
                quads_sdf_obs[i, g_id] = min_dist - obst_radius

    return quads_sdf_obs


@njit
def get_quad_circle_obst_collision_arr(quad_poses, obst_poses, obst_radius, quad_radius):
    quad_num = len(quad_poses)
    collide_threshold = quad_radius + obst_radius
    # Get distance matrix b/w quad and obst
    quad_collisions = -1 * np.ones(quad_num)
    for i, q_pos in enumerate(quad_poses):
        for j, o_pos in enumerate(obst_poses):
            dist = np.linalg.norm(q_pos - o_pos)
            if dist <= collide_threshold:
                quad_collisions[i] = j
                break

    return quad_collisions


@njit
def get_quad_grid_obst_collision_arr(quad_poses, obst_poses, obst_radius, quad_radius):
    quad_num = len(quad_poses)
    collide_threshold = quad_radius + obst_radius
    # Get distance matrix b/w quad and obst
    quad_collisions = -1 * np.ones(quad_num)
    for i, q_pos in enumerate(quad_poses):
        for j, o_pos in enumerate(obst_poses):
            if abs(q_pos[0] - o_pos[0]) <= collide_threshold and abs(q_pos[1] - o_pos[1]) <= collide_threshold:
                quad_collisions[i] = j
                break

    return quad_collisions


@njit
def get_cell_centers(obst_area_length, obst_area_width, grid_size=1.):
    count = 0
    i_len = obst_area_length / grid_size
    j_len = obst_area_width / grid_size
    cell_centers = np.zeros((int(i_len * j_len), 2))
    for i in np.arange(0, obst_area_length, grid_size):
        for j in np.arange(obst_area_width - grid_size, -grid_size, -grid_size):
            cell_centers[count][0] = i + (grid_size / 2) - obst_area_length // 2
            cell_centers[count][1] = j + (grid_size / 2) - obst_area_width // 2
            count += 1

    return cell_centers


def find_zero_chunks(grid):
    """ Find and return all connected chunks of 0s in the grid. """

    def dfs(x, y):
        """ Perform Depth-First Search to find connected cells. """
        if x < 0 or x >= N or y < 0 or y >= N or grid[x][y] == 1 or (x, y) in visited:
            return
        visited.add((x, y))
        chunk.append((x, y))
        for dx, dy in directions:
            dfs(x + dx, y + dy)

    N = len(grid)
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # Possible movements: up, down, left, right
    visited = set()
    chunks = []

    for i in range(N):
        for j in range(N):
            if grid[i][j] == 0 and (i, j) not in visited:
                chunk = []
                dfs(i, j)
                chunks.append(chunk)

    return chunks


def get_pos_arr(selected_indices, cell_centers, obst_area_length, room_height, grid_size):
    obst_pos_arr = []
    for obst_id in selected_indices:
        rid, cid = obst_id[0], obst_id[1]
        x, y = cell_centers[rid + int(obst_area_length / grid_size) * cid]
        obst_item = list((x, y))
        obst_item.append(room_height / 2.0)
        obst_pos_arr.append(obst_item)

    return obst_pos_arr


def visualize_chunks(grid, chunks, start, end):
    """ Visualize the grid with different chunks marked in different colors. """
    N = len(grid)
    color_grid = np.zeros((N, N))

    for color, chunk in enumerate(chunks, start=3):
        for x, y in chunk:
            color_grid[x][y] = color

    for start_item in start:
        color_grid[start_item[0]][start_item[1]] = 1  # Mark start with color 1
    for end_item in end:
        color_grid[end_item[0]][end_item[1]] = 2  # Mark end with the same color

    plt.figure(figsize=(8, 8))
    plt.imshow(color_grid, cmap='tab20', interpolation='nearest')
    plt.colorbar()
    plt.title('Grid Visualization with Different Chunks Colored')
    plt.show()


if __name__ == "__main__":
    from gym_art.quadrotor_multi.obstacles.test.unit_test import unit_test
    from gym_art.quadrotor_multi.obstacles.test.speed_test import speed_test

    # Unit Test
    unit_test()
    speed_test()
