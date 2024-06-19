from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription
from swarm_rl.runs.obstacles.single_drone.quad_obstacle_single_baseline import QUAD_BASELINE_CLI_1

_params = ParamGrid(
    [
        ("seed", [0000, 3333]),
        ("normalize_input", [True]),
        ("quads_obst_collision_smooth_max_penalty", [1.0, 2.0]),
        ("quads_obst_collision_smooth_penalty_range", [0.5, 1.0, 2.0]),
    ]
)

OBSTACLE_MODEL_CLI = QUAD_BASELINE_CLI_1 + (
    ' --quads_obst_grid_size=0.5 --quads_obst_spawn_center=False --quads_obst_grid_size_range 0.5 0.8 '
    '--quads_obstacle_tof_resolution=4 --quads_obst_grid_size_random=True '
    '--with_wandb=True --wandb_project=Quad-Swarm-RL --wandb_user=multi-drones '
    '--wandb_group=sd_mo_normal_input_search_obst_smooth'
)

_experiment = Experiment(
    "sd_mo_normal_input_search_obst_smooth",
    OBSTACLE_MODEL_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription("single_drone_multi_obst", experiments=[_experiment])