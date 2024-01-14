from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription
from swarm_rl.runs.hybrid.baseline import QUAD_BASELINE_CLI_8

_params = ParamGrid(
    [
        ("seed", [0000, 3333]),
        ("quads_max_obst_aggressive", [0.2]),
        ("quads_episode_duration", [15.0, 30.0, 60.0]),
    ]
)

OBSTACLE_MODEL_CLI = QUAD_BASELINE_CLI_8 + (
    # Self
    ' --quads_num_agents=32 --num_workers=18 --num_envs_per_worker=2 --quads_obs_repr=xyz_vxyz_R_omega_floor '
    '--quads_obs_acc_his=False --quads_cost_enable_extra=False --train_for_env_steps=2000000000 '
    # Obstacle
    '--quads_obst_density=0.8 --quads_obst_size=0.85 --quads_obst_gap=-1 '
    # Cost
    '--quads_cost_rl_mellinger=0.0 --quads_cost_extra_rl_real=0.0 --quads_cost_rl_sbc=0.1 '
    '--quads_sbc_boundary=0.1 --quads_cost_act_change=0.0 --quads_cost_no_sol=0.0 --quads_cost_cbf_agg=0.0 '
    # SBC
    '--quads_enable_sbc=True --quads_sbc_radius=0.05 --quads_max_acc=4.0 --quads_max_room_aggressive=0.2 '
    '--quads_neighbor_range=2.0 --quads_obst_range=2.0 --quads_max_neighbor_aggressive=0.2 '
    # Domain Randomization
    '--quads_domain_random=False --quads_obst_density_min=0.2 --quads_obst_density_max=0.8 '
    '--quads_obst_gap_min=0.12 --quads_obst_gap_max=0.28 '
    # Annealing
    '--anneal_collision_steps=0 '
    # Safe Annealing
    '--quads_anneal_safe_start_steps=0 --quads_anneal_safe_total_steps=0 --cbf_agg_anneal_steps=0 '
    # Wandb
    '--with_wandb=True --wandb_project=Quad-Hybrid --wandb_user=multi-drones '
    '--wandb_group=hybrid_compare_baseline_os_0.85'
)

_experiment = Experiment(
    "hybrid_compare_baseline_os_0.85",
    OBSTACLE_MODEL_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription("hybrid", experiments=[_experiment])
