import random
from collections import deque
from copy import deepcopy

import gymnasium as gym
import numpy as np

from sample_factory.utils.utils import log


class ReplayBufferEvent:
    def __init__(self, env, obs):
        self.env = env
        self.obs = obs
        self.num_replayed = 0


class ReplayBuffer:
    def __init__(self, control_frequency, cp_step_size=0.5, buffer_size=20):
        self.control_frequency = control_frequency
        self.cp_step_size_sec = cp_step_size  # how often (seconds) a checkpoint is saved
        self.cp_step_size_freq = self.cp_step_size_sec * self.control_frequency
        self.buffer = deque([], maxlen=buffer_size)

    def get_buffer_id(self):
        max_replayed_id = 0
        max_replayed_num = 0
        for i in range(len(self.buffer)):
            cur_num_replayed = self.buffer[i].num_replayed
            if cur_num_replayed > max_replayed_num:
                max_replayed_id = i
                max_replayed_num = cur_num_replayed
        return max_replayed_id

    def write_cp_to_buffer(self, env, obs):
        """
        A collision was found, and we want to load the corresponding checkpoint from X seconds ago into the buffer to be sampled later on
        """
        env.saved_in_replay_buffer = True

        # For example, replace the item with the lowest number of collisions in the last 10 replays
        evt = ReplayBufferEvent(env, obs)
        if len(self.buffer) < self.buffer.maxlen:
            self.buffer.append(evt)
            add_pos_id = len(self.buffer)
        else:
            buffer_id = self.get_buffer_id()
            self.buffer[buffer_id] = evt
            add_pos_id = buffer_id

        log.info('Added new collision event to buffer at: %s', str(add_pos_id))

    def sample_event(self):
        """
        Sample an event to replay
        """
        idx = random.randint(0, len(self.buffer) - 1)
        log.info('Replaying event at idx: %s', str(idx))
        self.buffer[idx].num_replayed += 1
        return self.buffer[idx]

    def cleanup(self):
        new_buffer = deque([], maxlen=self.buffer.maxlen)
        for event in self.buffer:
            if event.num_replayed < 10:
                new_buffer.append(event)

        self.buffer = new_buffer

    def avg_num_replayed(self):
        replayed_stats = [e.num_replayed for e in self.buffer]
        if not replayed_stats:
            return 0
        return np.mean(replayed_stats)

    def __len__(self):
        return len(self.buffer)


class ExperienceReplayWrapper(gym.Wrapper):
    def __init__(self, env, replay_buffer_sample_prob, init_obst_params, domain_random=False, dr_params=None):
        super().__init__(env)
        self.replay_buffer = ReplayBuffer(control_frequency=env.envs[0].control_freq)
        self.replay_buffer_sample_prob = replay_buffer_sample_prob

        # Default parameters for obstacles, used when domain_random=False
        # Including obstacle size, obstacle density, obstacle gap and number of obstacles
        self.curr_obst_params = init_obst_params

        # Domain randomization
        self.domain_random = domain_random
        if self.domain_random:
            self.dr_params = dr_params

        # keep only checkpoints from the last 3 seconds
        self.max_episode_checkpoints_to_keep = int(3.0 / self.replay_buffer.cp_step_size_sec)
        self.episode_checkpoints = deque([], maxlen=self.max_episode_checkpoints_to_keep)

        self.save_time_before_collision_sec = 1.5
        self.last_tick_added_to_buffer = -1e9

        # variables for tensorboard
        self.replayed_events = 0
        self.episode_counter = 0

    def save_checkpoint(self, obs):
        """
        Save a checkpoint every X steps so that we may load it later if a collision was found. This is NOT the same as the buffer
        Checkpoints are added to the buffer only if we find a collision and want to replay that event later on
        """
        self.episode_checkpoints.append((deepcopy(self.env), deepcopy(obs)))

    def sample_dr_params(self):
        dr_params = {}
        for k in self.dr_params:
            dr_params[k] = np.random.choice(self.dr_params[k])

        return dr_params

    def reset(self):
        """For reset we just use the default implementation. This reset actually never called"""
        dr_params = None
        # If using domain randomization, sample new parameters and reset the environment with them
        if self.domain_random:
            dr_params = self.sample_dr_params()
            self.curr_obst_params = dr_params
        log.info('Current obstacle params %s', self.curr_obst_params)
        return self.env.reset(dr_params)

    def step(self, action):
        obs, rewards, dones, infos = self.env.step(action)

        if any(dones):
            not_in_replay_buffer_bool = not self.env.saved_in_replay_buffer
            reached_goal_threshold_bool = self.env.replay_reached_goal_ratio >= 0.6
            if not_in_replay_buffer_bool and reached_goal_threshold_bool and len(self.env.replay_distance_to_goal[0]) > 0:
                ctrl_freq = self.env.envs[0].control_freq
                self.env.replay_distance_to_goal = np.array(self.env.replay_distance_to_goal)
                dist_1s = ctrl_freq * np.mean(self.env.replay_distance_to_goal[:, int(-1 * ctrl_freq):])
                if dist_1s >= self.env.replay_goal_reach_metric:
                    steps_ago = min(self.max_episode_checkpoints_to_keep, len(self.episode_checkpoints))
                    try:
                        env, obs = self.episode_checkpoints[-steps_ago]
                    except:
                        log.info('len(self.episode_checkpoints): %s', str(len(self.episode_checkpoints)))
                        env, obs = self.episode_checkpoints[-len(self.episode_checkpoints)]

                    self.replay_buffer.write_cp_to_buffer(env, obs)

            obs = self.new_episode()
            for i in range(len(infos)):
                if not infos[i]["episode_extra_stats"]:
                    infos[i]["episode_extra_stats"] = dict()

                tag = "replay"
                infos[i]["episode_extra_stats"].update({
                    f"{tag}/replay_rate": self.replayed_events / self.episode_counter,
                    f"{tag}/new_episode_rate": (self.episode_counter - self.replayed_events) / self.episode_counter,
                    f"{tag}/replay_buffer_size": len(self.replay_buffer),
                    f"{tag}/avg_replayed": self.replay_buffer.avg_num_replayed(),
                })
        else:
            enable_replay_buffer_bool = self.env.use_replay_buffer and self.env.activate_replay_buffer
            not_in_replay_buffer_bool = not self.env.saved_in_replay_buffer
            save_period_bool = self.env.envs[0].tick % self.replay_buffer.cp_step_size_freq == 0
            if enable_replay_buffer_bool and not_in_replay_buffer_bool and save_period_bool:
                self.save_checkpoint(obs)

            collision_flag = self.env.last_step_unique_collisions.any()
            if self.env.use_obstacles:
                collision_flag = collision_flag or len(self.env.curr_quad_col) > 0

            no_sol_flag = False
            if self.env.enable_sbc:
                no_sol_flag = self.env.no_sol_flag

            add_ent_flag = collision_flag or no_sol_flag

            grace_tick = self.env.collisions_grace_period_seconds * self.env.envs[0].control_freq
            out_grace_bool = self.env.envs[0].tick > grace_tick
            out_add_gap_bool = self.env.envs[0].tick - self.last_tick_added_to_buffer > 2 * self.env.envs[0].control_freq
            if add_ent_flag and enable_replay_buffer_bool and not_in_replay_buffer_bool and out_grace_bool and out_add_gap_bool:
                # added this check to avoid adding a lot of collisions from the same episode to the buffer
                steps_ago = int(self.save_time_before_collision_sec / self.replay_buffer.cp_step_size_sec)
                steps_ago = min(steps_ago, len(self.episode_checkpoints))

                env, obs = self.episode_checkpoints[-steps_ago]
                self.replay_buffer.write_cp_to_buffer(env, obs)
                # this allows us to add a copy of this episode to the buffer once again
                # if another collision happens
                self.env.collision_occurred = False

                self.last_tick_added_to_buffer = self.env.envs[0].tick

        return obs, rewards, dones, infos

    def new_episode(self):
        """
        Normally this would go into reset(), but MultiQuadEnv is a multi-agent env that automatically resets.
        This means that reset() is never actually called externally and we need to take care of starting our new episode.
        """
        self.episode_counter += 1
        self.last_tick_added_to_buffer = -1e9
        self.episode_checkpoints = deque([], maxlen=self.max_episode_checkpoints_to_keep)

        replay_bool = np.random.uniform(low=0, high=1) < self.replay_buffer_sample_prob
        activate_bool = self.env.activate_replay_buffer and len(self.replay_buffer) > 0
        if replay_bool and activate_bool:
            self.replayed_events += 1
            event = self.replay_buffer.sample_event()
            env = event.env
            obs = event.obs
            replayed_env = deepcopy(env)
            replayed_env.scenes = self.env.scenes
            self.curr_obst_params = replayed_env.obst_params

            # we want to use these for tensorboard, so reset them to zero to get accurate stats
            replayed_env.collisions_per_episode = replayed_env.collisions_after_settle = 0
            replayed_env.obst_quad_collisions_per_episode = replayed_env.obst_quad_collisions_after_settle = 0
            self.env = replayed_env

            self.replay_buffer.cleanup()

            return obs

        else:
            dr_params = None
            if self.domain_random:
                dr_params = self.sample_dr_params()
                self.curr_obst_params = dr_params
            obs = self.env.reset(dr_params)

            self.env.saved_in_replay_buffer = False
            return obs
