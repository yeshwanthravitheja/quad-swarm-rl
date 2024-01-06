python -m swarm_rl.enjoy \
--algo=APPO \
--env=quadrotor_multi \
--replay_buffer_sample_prob=0 \
--quads_use_numba=True \
--train_dir=/home/zhehui/quad-swarm-rl/train_dir \
--experiment=01_grid_search_no_sol_rl_v1_see_0_q.m.n.agg_50.0_q.m.o.agg_25.0_q.m.acc_4.0 \
--quads_episode_duration=15.0 --quads_view_mode=global --anneal_collision_steps=0 --quads_anneal_safe_start_steps=0 \
--quads_anneal_safe_total_steps=0 --cbf_agg_anneal_steps=0 --quads_mode=mix --quads_render=False --no_render \
--quads_domain_random=True --quads_obst_size=0.6 \
--quads_obst_density_min=0.2 --quads_obst_density_max=0.8 --quads_obst_gap_min=0.15 --quads_obst_gap_max=0.4 \
--quads_max_neighbor_aggressive=0.1 --quads_max_obst_aggressive=0.1