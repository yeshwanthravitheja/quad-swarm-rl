
import numpy as np
import copy

from gym_art.quadrotor_multi.scenarios.obstacles.o_base import Scenario_o_base
from gym_art.quadrotor_multi.quadrotor_traj_gen import QuadTrajGen
from gym_art.quadrotor_multi.quadrotor_planner import traj_eval

class Scenario_o_random_dynamic_goal(Scenario_o_base):
    def __init__(self, quads_mode, envs, num_agents, room_dims):
        super().__init__(quads_mode, envs, num_agents, room_dims)
        self.approch_goal_metric = 0.5
        self.goal_generator = []
        
    def update_formation_size(self, new_formation_size):
        pass

    def step(self):
        # self.update_formation_and_relate_param()

        tick = self.envs[0].tick
        time = tick*self.envs[0].dt # This should be in seconds
        
        for i in range(self.num_agents):
            next_goal = self.goal_generator[i].piecewise_eval(time)
            next_goal = next_goal.as_nparray()
            self.end_point[i] = next_goal[:3]
            print(time, self.end_point[i])
            self.goals = copy.deepcopy(self.end_point)
            
        for i, env in enumerate(self.envs):
            env.goal = self.end_point[i]

        if tick <= self.duration_step:
            return

        self.duration_step += int(self.envs[0].ep_time * self.envs[0].control_freq)

        return

    def reset(self, obst_map, cell_centers):        
        self.obstacle_map = obst_map
        self.cell_centers = cell_centers
        if obst_map is None:
            raise NotImplementedError

        obst_map_locs = np.where(self.obstacle_map == 0)
        self.free_space = list(zip(*obst_map_locs))
        
        self.duration_step = int(np.random.uniform(low=2.0, high=4.0) * self.envs[0].control_freq)

        self.start_point = []
        self.end_point = []
        for i in range(self.num_agents):
            self.start_point.append(self.generate_pos_obst_map())
            initial_state = traj_eval()
            initial_state.set_initial_pos(self.start_point[i])
            
            # self.end_point.append(self.generate_pos_obst_map())
            
            self.goal_generator.append(QuadTrajGen(poly_degree=7))
            # duration = self.envs[0].ep_time  # Seconds
            duration = self.duration_step*self.envs[i].dt
            print("TRAJ DURATION: ", duration)
            self.goal_generator[i].plan_go_to_from(initial_state=initial_state, desired_state=np.append(self.generate_pos_obst_map(), 0), duration=duration, current_time=0)
            
            #Find the initial goal
            # self.end_point.append(self.goal_generator[i].piecewise_eval(0.001).pos)
            # print("Quad Initial Goal:", self.end_point[i])
            self.end_point.append(self.start_point[i])

        print("Quadrotor Start: ", self.start_point)
        print("Quadrotor End: ", self.end_point)
        
        # self.start_point = self.generate_pos_obst_map_2(num_agents=self.num_agents)
        # self.end_point = self.generate_pos_obst_map_2(num_agents=self.num_agents)
        # self.start_point = self.generate_pos_obst_map_2(self.num_agents)
        # self.end_point = self.generate_pos_obst_map_2(self.num_agents)

        # self.update_formation_and_relate_param()

        self.formation_center = np.array((0., 0., 2.))
        self.spawn_points = copy.deepcopy(self.start_point)
        
        self.goals = copy.deepcopy(self.end_point)