from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription
from swarm_rl.runs.multi_quad.quad_multi_baseline import MULTI_QUAD_BASELINE_CLI

_params = ParamGrid(
    [
        ("seed", [0000, 1111, 2222, 3333]),
        ("quads_num_agents", [8]),
    ]
)

OBSTACLE_MODEL_CLI = MULTI_QUAD_BASELINE_CLI + (
    ' --num_workers=36 --num_envs_per_worker=4 --rnn_size=16 --quads_obs_repr=xyz_vxyz_R_omega '
    '--quads_neighbor_hidden_size=8 --quads_neighbor_visible_num=2 --quads_neighbor_encoder_type=mean_embed '
    '--with_wandb=True --wandb_project=Quad-Swarm-RL --wandb_user=multi-drones '
    '--wandb_group=multi_drone_no_obst_mean_embed'
)

_experiment = Experiment(
    "multi_drone_no_obst_mean_embed",
    OBSTACLE_MODEL_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription("multi_drone", experiments=[_experiment])