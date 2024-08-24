python -m swarm_rl.train \
--env=quadrotor_multi \
--train_for_env_steps=1000000000 \
--algo=APPO \
--use_rnn=False \
--num_workers=4 \
--num_envs_per_worker=4 \
--learning_rate=0.0001 \
--ppo_clip_value=5.0 \
--recurrence=1 \
--nonlinearity=tanh \
--actor_critic_share_weights=False \
--policy_initialization=xavier_uniform \
--adaptive_stddev=False \
--with_vtrace=False \
--max_policy_lag=100000000 \
--rnn_size=256 \
--gae_lambda=1.00 \
--max_grad_norm=5.0 \
--exploration_loss_coeff=0.0 \
--rollout=128 \
--batch_size=1024 \
--with_pbt=False \
--normalize_input=False \
--normalize_returns=False \
--reward_clip=10 \
--quads_use_numba=True \
--save_milestones_sec=3600 \
--anneal_collision_steps=300000000 \
--replay_buffer_sample_prob=0.75 \
--quads_mode=mix \
--quads_episode_duration=15.0 \
--quads_obs_repr=xyz_vxyz_R_omega \
--quads_neighbor_hidden_size=256 \
--quads_neighbor_obs_type=pos_vel \
--quads_collision_hitbox_radius=2.0 \
--quads_collision_falloff_radius=4.0 \
--quads_collision_reward=5.0 \
--quads_collision_smooth_max_penalty=10.0 \
--quads_neighbor_encoder_type=attention \
--quads_neighbor_visible_num=6 \
--quads_use_obstacles=False \
--quads_use_downwash=True \
--experiment=test_multi_drone


: << 'COMMENT'

# This script contains command-line arguments for training a quadrotor swarm reinforcement learning model.

# Override default value for the number of quadrotors
# Usage: --quads_num_agents <int>
# Default: 8

# Choose the observation space representation for quadrotor self
# Usage: --quads_obs_repr <str>
# Choices: ['xyz_vxyz_R_omega', 'xyz_vxyz_R_omega_floor', 'xyz_vxyz_R_omega_wall']
# Default: 'xyz_vxyz_R_omega'

# Override default value for episode duration
# Usage: --quads_episode_duration <float>
# Default: 15.0

# Choose the type of the neighborhood encoder
# Usage: --quads_encoder_type <str>
# Default: "corl"

# Number of neighbors to consider
# Usage: --quads_neighbor_visible_num <int>
# Choices: -1 (all), 0 (blind agents), 0 < n < num_agents-1 (nonzero number of agents)
# Default: -1

# Choose what kind of observation to send to the encoder for neighbors
# Usage: --quads_neighbor_obs_type <str>
# Choices: ['none', 'pos_vel']
# Default: 'none'

# The hidden size for the neighbor encoder
# Usage: --quads_neighbor_hidden_size <int>
# Default: 256

# The type of the neighborhood encoder
# Usage: --quads_neighbor_encoder_type <str>
# Choices: ['attention', 'mean_embed', 'mlp', 'no_encoder']
# Default: 'attention'

# Override default value for quadcol_bin reward, which means collisions between quadrotors
# Usage: --quads_collision_reward <float>
# Default: 0.0

# When the distance between two drones is less than N arm_length, we would view them as collide
# Usage: --quads_collision_hitbox_radius <float>
# Default: 2.0

# The falloff radius for the smooth penalty. -1.0: no smooth penalty
# Usage: --quads_collision_falloff_radius <float>
# Default: -1.0

# The upper bound of the collision function given distance among drones
# Usage: --quads_collision_smooth_max_penalty <float>
# Default: 10.0

# Use obstacles or not
# Usage: --quads_use_obstacles <bool>
# Default: False

# Choose what kind of observation to send to the encoder for obstacles
# Usage: --quads_obstacle_obs_type <str>
# Choices: ['none', 'octomap']
# Default: 'none'

# Obstacle density in the map
# Usage: --quads_obst_density <float>
# Default: 0.2

# The radius of obstacles
# Usage: --quads_obst_size <float>
# Default: 1.0

# The spawning area of obstacles
# Usage: --quads_obst_spawn_area <float> <float>
# Default: [6.0, 6.0]

# Use domain randomization or not
# Usage: --quads_domain_random <bool>
# Default: False

# Enable obstacle density randomization or not
# Usage: --quads_obst_density_random <bool>
# Default: False

# The minimum obstacle density when enabling domain randomization
# Usage: --quads_obst_density_min <float>
# Default: 0.05

# The maximum obstacle density when enabling domain randomization
# Usage: --quads_obst_density_max <float>
# Default: 0.2

# Enable obstacle size randomization or not
# Usage: --quads_obst_size_random <bool>
# Default: False

# The minimum obstacle size when enabling domain randomization
# Usage: --quads_obst_size_min <float>
# Default: 0.3

# The maximum obstacle size when enabling domain randomization
# Usage: --quads_obst_size_max <float>
# Default: 0.6

# The hidden size for the obstacle encoder
# Usage: --quads_obst_hidden_size <int>
# Default: 256

# The type of the obstacle encoder
# Usage: --quads_obst_encoder_type <str>
# Default: 'mlp'

# Override default value for quadcol_bin_obst reward, which means collisions between quadrotor and obstacles
# Usage: --quads_obst_collision_reward <float>
# Default: 0.0

# Apply downwash or not
# Usage: --quads_use_downwash <bool>
# Default: False

# Whether to use numba for jit or not
# Usage: --quads_use_numba <bool>
# Default: False

# Choose which scenario to run
# Usage: --quads_mode <str>
# Choices: ['static_same_goal', 'static_diff_goal', 'dynamic_same_goal', 'dynamic_diff_goal', 'ep_lissajous3D', 'ep_rand_bezier', 'swarm_vs_swarm', 'swap_goals', 'dynamic_formations', 'mix', 'o_uniform_same_goal_spawn', 'o_random', 'o_dynamic_diff_goal', 'o_dynamic_same_goal', 'o_diagonal', 'o_static_same_goal', 'o_static_diff_goal', 'o_swap_goals', 'o_ep_rand_bezier']
# Default: 'static_same_goal'

# Length, width, and height dimensions respectively of the quadrotor environment
# Usage: --quads_room_dims <float> <float> <float>
# Default: [10., 10., 10.]

# Probability at which we sample from the replay buffer rather than resetting the environment
# Usage: --replay_buffer_sample_prob <float>
# Default: 0.0

# Anneal collision penalties over this many steps. Default (0.0) is no annealing
# Usage: --anneal_collision_steps <float>
# Default: 0.0

# Choose which kind of view/camera to use
# Usage: --quads_view_mode <str> <str> ...
# Choices: ['topdown', 'chase', 'side', 'global', 'corner0', 'corner1', 'corner2', 'corner3', 'topdownfollow']
# Default: ['topdown', 'chase', 'global']

# Use render or not
# Usage: --quads_render <bool>
# Default: False

# Visualize v value map
# Usage: --visualize_v_value

# Whether to use sim2real or not
# Usage: --quads_sim2real <bool>
# Default: False
 p.add_argument('--quads_num_agents', default=8, type=int, help='Override default value for the number of quadrotors')
    p.add_argument('--quads_obs_repr', default='xyz_vxyz_R_omega', type=str,
                   choices=['xyz_vxyz_R_omega', 'xyz_vxyz_R_omega_floor', 'xyz_vxyz_R_omega_wall'],
                   help='obs space for quadrotor self')
    p.add_argument('--quads_episode_duration', default=15.0, type=float,
                   help='Override default value for episode duration')
    p.add_argument('--quads_encoder_type', default="corl", type=str, help='The type of the neighborhood encoder')

    # Neighbor
    # Neighbor Features
    p.add_argument('--quads_neighbor_visible_num', default=-1, type=int, help='Number of neighbors to consider. -1=all '
                                                                          '0=blind agents, '
                                                                          '0<n<num_agents-1 = nonzero number of agents')
    p.add_argument('--quads_neighbor_obs_type', default='none', type=str,
                   choices=['none', 'pos_vel'], help='Choose what kind of obs to send to encoder.')

    # # Neighbor Encoder
    p.add_argument('--quads_neighbor_hidden_size', default=256, type=int,
                   help='The hidden size for the neighbor encoder')
    p.add_argument('--quads_neighbor_encoder_type', default='attention', type=str,
                   choices=['attention', 'mean_embed', 'mlp', 'no_encoder'],
                   help='The type of the neighborhood encoder')

    # # Neighbor Collision Reward
    p.add_argument('--quads_collision_reward', default=0.0, type=float,
                   help='Override default value for quadcol_bin reward, which means collisions between quadrotors')
    p.add_argument('--quads_collision_hitbox_radius', default=2.0, type=float,
                   help='When the distance between two drones are less than N arm_length, we would view them as '
                        'collide.')
    p.add_argument('--quads_collision_falloff_radius', default=-1.0, type=float,
                   help='The falloff radius for the smooth penalty. -1.0: no smooth penalty')
    p.add_argument('--quads_collision_smooth_max_penalty', default=10.0, type=float,
                   help='The upper bound of the collision function given distance among drones')

    # Obstacle
    # # Obstacle Features
    p.add_argument('--quads_use_obstacles', default=False, type=str2bool, help='Use obstacles or not')
    p.add_argument('--quads_obstacle_obs_type', default='none', type=str,
                   choices=['none', 'octomap'], help='Choose what kind of obs to send to encoder.')
    p.add_argument('--quads_obst_density', default=0.2, type=float, help='Obstacle density in the map')
    p.add_argument('--quads_obst_size', default=1.0, type=float, help='The radius of obstacles')
    p.add_argument('--quads_obst_spawn_area', nargs='+', default=[6.0, 6.0], type=float,
                   help='The spawning area of obstacles')
    p.add_argument('--quads_domain_random', default=False, type=str2bool, help='Use domain randomization or not')
    p.add_argument('--quads_obst_density_random', default=False, type=str2bool, help='Enable obstacle density randomization or not')
    p.add_argument('--quads_obst_density_min', default=0.05, type=float,
                   help='The minimum of obstacle density when enabling domain randomization')
    p.add_argument('--quads_obst_density_max', default=0.2, type=float,
                   help='The maximum of obstacle density when enabling domain randomization')
    p.add_argument('--quads_obst_size_random', default=False, type=str2bool, help='Enable obstacle size randomization or not')
    p.add_argument('--quads_obst_size_min', default=0.3, type=float,
                   help='The minimum obstacle size when enabling domain randomization')
    p.add_argument('--quads_obst_size_max', default=0.6, type=float,
                   help='The maximum obstacle size when enabling domain randomization')

    # # Obstacle Encoder
    p.add_argument('--quads_obst_hidden_size', default=256, type=int, help='The hidden size for the obstacle encoder')
    p.add_argument('--quads_obst_encoder_type', default='mlp', type=str, help='The type of the obstacle encoder')

    # # Obstacle Collision Reward
    p.add_argument('--quads_obst_collision_reward', default=0.0, type=float,
                   help='Override default value for quadcol_bin_obst reward, which means collisions between quadrotor '
                        'and obstacles')

    # Aerodynamics
    # # Downwash
    p.add_argument('--quads_use_downwash', default=False, type=str2bool, help='Apply downwash or not')

    # Numba Speed Up
    p.add_argument('--quads_use_numba', default=False, type=str2bool, help='Whether to use numba for jit or not')

    # Scenarios
    p.add_argument('--quads_mode', default='static_same_goal', type=str,
                   choices=['static_same_goal', 'static_diff_goal', 'dynamic_same_goal', 'dynamic_diff_goal',
                            'ep_lissajous3D', 'ep_rand_bezier', 'swarm_vs_swarm', 'swap_goals', 'dynamic_formations',
                            'mix', 'o_uniform_same_goal_spawn', 'o_random',
                            'o_dynamic_diff_goal', 'o_dynamic_same_goal', 'o_diagonal', 'o_static_same_goal',
                            'o_static_diff_goal', 'o_swap_goals', 'o_ep_rand_bezier'],
                   help='Choose which scenario to run. ep = evader pursuit')

    # Room
    p.add_argument('--quads_room_dims', nargs='+', default=[10., 10., 10.], type=float,
                   help='Length, width, and height dimensions respectively of the quadrotor env')

    # Replay Buffer
    p.add_argument('--replay_buffer_sample_prob', default=0.0, type=float,
                   help='Probability at which we sample from it rather than resetting the env. Set to 0.0 (default) '
                        'to disable the replay. Set to value in (0.0, 1.0] to use replay buffer')

    # Annealing
    p.add_argument('--anneal_collision_steps', default=0.0, type=float, help='Anneal collision penalties over this '
                                                                             'many steps. Default (0.0) is no '
                                                                             'annealing')

    # Rendering
    p.add_argument('--quads_view_mode', nargs='+', default=['topdown', 'chase', 'global'],
                   type=str, choices=['topdown', 'chase', 'side', 'global', 'corner0', 'corner1', 'corner2', 'corner3', 'topdownfollow'],
                   help='Choose which kind of view/camera to use')
    p.add_argument('--quads_render', default=False, type=bool, help='Use render or not')
    p.add_argument('--visualize_v_value', action='store_true', help="Visualize v value map")

    # Sim2Real
    p.add_argument('--quads_sim2real', default=False, type=str2bool, help='Whether to use sim2real or not')

COMMENT
