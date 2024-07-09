
import numpy as np
import copy

from gym_art.quadrotor_multi.scenarios.obstacles.o_base import Scenario_o_base
from gym_art.quadrotor_multi.quadrotor_traj_gen import QuadTrajGen
from gym_art.quadrotor_multi.quadrotor_planner import traj_eval
from sample_factory.envs.env_utils import TrainingInfoInterface


class Scenario_o_random_dynamic_goal_curriculum(Scenario_o_base, TrainingInfoInterface):
    """ This scenario implements a 13 dim goal that tracks a smooth polynomial trajectory. 
        Each goal point is evaluated through the polynomial generated per reset. This specific
        implementation increases the number of polynomial evaluations by the success of training."""
        
    def __init__(self, quads_mode, envs, num_agents, room_dims):
        super().__init__(quads_mode, envs, num_agents, room_dims)
        TrainingInfoInterface.__init__(self)
        self.approch_goal_metric = 0.5
        self.goal_generator = []

        # This value sets how many TOTAL goal points are to be evaluated during an episode. This does not include
        # the initial hover start point.
        self.goal_curriculum = [1.0] * self.num_agents

        #Tracks the required time between shifts in goal.
        self.goal_dt = [0] * self.num_agents

        #Tracks the current time before a goal is changed. Init to all zeros.
        self.goal_time = [0] * self.num_agents

        #Tracks whether curriculum has started. When curriculum starts, there may be a chance the drones performance drops. 
        # In this case, we want to force curriculum.
        self.begin_curriculum = False

    def __round_dt(x, prec=2, base=self.envs[0].sim_steps*self.envs[0].dt):
        """ Rounds to nearest decimal with precision."""
        return round(base * round(float(x)/base),prec)

        
    def update_formation_size(self, new_formation_size):
        pass

    def step(self):
        self.update_formation_and_relate_param()

        tick = self.envs[0].tick
        
        time = self.envs[0].sim_steps*tick*(self.envs[0].dt) #  Current time in seconds.
        
        for i in range(self.num_agents):
            # self.goal_time[i] += self.envs[0].sim_steps*self.envs[0].dt

            # change goals if we are within 1 time step
            # if (abs(self.goal_time[i] - self.goal_dt[i]) < (self.envs[0].sim_steps*self.envs[0].dt)):
            if (time % self.goal_dt[i] == 0):

                next_goal = self.goal_generator[i].piecewise_eval(time)
        
                self.end_point[i] = next_goal.as_nparray()

                self.goals = copy.deepcopy(self.end_point)

                # self.goal_time[i] = 0
            
        for i, env in enumerate(self.envs):
            env.goal = self.end_point[i]
        
        return

    def reset(self, obst_map, cell_centers, distance_to_goal_metric, curriculum_episdode_count): 
  
        self.obstacle_map = obst_map
        self.cell_centers = cell_centers
        
        if obst_map is None:
            raise NotImplementedError

        obst_map_locs = np.where(self.obstacle_map == 0)
        self.free_space = list(zip(*obst_map_locs))

        self.start_point = []
        self.end_point = []
        for i in range(self.num_agents):
            self.start_point.append(self.generate_pos_obst_map())
            
            initial_state = traj_eval()
            initial_state.set_initial_pos(self.start_point[i])
            
            self.goal_generator.append(QuadTrajGen(poly_degree=7))
            
            final_goal = self.generate_pos_obst_map()
            
            # Fix the goal height at 0.65 m
            final_goal[2] = 0.65
            traj_duration = np.random.uniform(low=2, high=self.envs[0].ep_time)

            # Generate trajectory with random time from (2, ep_time)
            self.goal_generator[i].plan_go_to_from(initial_state=initial_state, desired_state=np.append(final_goal, np.random.uniform(low=0, high=3.14)), 
                                                   duration=traj_duration, current_time=0)
            

            approx_total_training_steps = self.training_info.get('approx_total_training_steps', 0)

            print(approx_total_training_steps)
            # EPISODE BASED GOAL CURRICULUM
            if (len(distance_to_goal_metric[i]) == 10):
                avg_distance = sum(distance_to_goal_metric[i]) / len(distance_to_goal_metric[i])

                # We only start curriculum when the drone gets an average of 1 meter within the goal for the past 10 episodes.
                if (avg_distance <= 1.0) or (self.begin_curriculum):

                    # When the avg distance becomes less than 1, we want to ALWAYS use curriculum.
                    # Even if the drones performance drops in the beginning of curriculum
                    self.begin_curriculum = True

                    if ((curriculum_episdode_count % 10) == 0):
                        self.goal_curriculum[i] += 1
           
            # DISTANCE BASED GOAL CURRICULUM
            # Only start calculating curriculum if we have 10 policy rollouts.
            # if (len(distance_to_goal_metric[i]) == 10):
            #     avg_distance = sum(distance_to_goal_metric[i]) / len(distance_to_goal_metric[i])            
                
            #     # We only start curriculum when the drone gets an average of 1 meter within the goal.
            #     if (avg_distance <= 1.0):
                    
            #         #Change number of goal points based on average distance to goal for recent rollouts.
            #         self.goal_curriculum[i] = int(1 / (avg_distance)^100)
                    
            self.goal_dt[i] = self.__round_dt(traj_duration / self.goal_curriculum[i])

            #Find the initial goal
            if (self.begin_curriculum):
                self.end_point.append(self.goal_generator[i].piecewise_eval(0).as_nparray())
            else:
                self.end_point.append(self.goal_generator[i].piecewise_eval(self.envs[i].ep_time).as_nparray())
        
        self.update_formation_and_relate_param()

        self.formation_center = np.array((0., 0., 2.))
        self.spawn_points = copy.deepcopy(self.start_point)
        
        self.goals = copy.deepcopy(self.end_point)
        
        
        