from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription
from swarm_rl.runs.hybrid.baseline import QUAD_BASELINE_CLI_8

_params = ParamGrid(
    [
        ("seed", [0000, 3333]),
        ("quads_domain_random", [True]),
        ("quads_cost_cbf_agg", [0.05, 0.1, 0.2]),
    ]
)

OBSTACLE_MODEL_CLI = QUAD_BASELINE_CLI_8 + (
    # Self
    ' --quads_num_agents=8 --quads_obs_repr=xyz_vxyz_R_omega_floor --quads_episode_duration=15.0 '
    '--quads_obs_acc_his=False --quads_cost_enable_extra=False --train_for_env_steps=2000000000 '
    # Obstacle
    '--quads_obst_density=0.8 --quads_obst_size=0.85 --quads_obst_gap=0.15 '
    # Cost
    '--quads_cost_rl_mellinger=0.0 --quads_cost_extra_rl_real=0.0 --quads_cost_rl_sbc=0.1 '
    '--quads_sbc_boundary=0.1 --quads_cost_act_change=0.0 '
    # SBC
    '--quads_enable_sbc=True --quads_sbc_radius=0.05 --quads_max_neighbor_aggressive=50.0 '
    '--quads_max_obst_aggressive=25.0 --quads_max_acc=4.0 --quads_max_room_aggressive=1.0 '
    '--quads_neighbor_range=2.0 --quads_obst_range=2.0 '
    '--quads_obst_density_min=0.2 --quads_obst_density_max=0.8 --quads_obst_gap_min=0.15 --quads_obst_gap_max=0.4 '
    '--quads_obst_size=0.6 '
    # Annealing
    '--anneal_collision_steps=0 '
    # Safe Annealing
    '--quads_anneal_safe_start_steps=0 --quads_anneal_safe_total_steps=0 --cbf_agg_anneal_steps=0 '
    # Wandb
    '--with_wandb=True --wandb_project=Quad-Hybrid --wandb_user=multi-drones '
    '--wandb_group=domain_random_v2'
)

_experiment = Experiment(
    "domain_random_v2",
    OBSTACLE_MODEL_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription("hybrid", experiments=[_experiment])
