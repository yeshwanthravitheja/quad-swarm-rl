from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription
from swarm_rl.runs.obstacles.quad_obstacle_baseline import QUAD_BASELINE_CLI_8

_params = ParamGrid(
    [
        ("seed", [0000, 1111, 2222, 3333]),
        ("quads_num_agents", [1]),
        ("quads_obstacle_tof_resolution", [4, 8]),
    ]
)

OBSTACLE_MODEL_CLI = QUAD_BASELINE_CLI_8 + (
    ' --num_envs_per_worker=32 --rnn_size=16 --quads_obs_repr=xyz_vxyz_R_omega --quads_obs_rel_rot=True '
    '--quads_neighbor_visible_num=0 --quads_neighbor_obs_type=none --quads_neighbor_hidden_size=16 '
    '--quads_obst_hidden_size=8 --quads_obst_density=0.2 --quads_obstacle_obs_type=ToFs '
    '--with_wandb=True --wandb_project=Quad-Swarm-RL --wandb_user=multi-drones '
    '--wandb_group=one_drone_search_tof_resolution'
)

_experiment = Experiment(
    "one_drone_search_tof_resolution",
    OBSTACLE_MODEL_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription("single_drone_multi_obst", experiments=[_experiment])