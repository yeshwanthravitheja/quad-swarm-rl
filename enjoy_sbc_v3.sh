python -m swarm_rl.enjoy \
--algo=APPO --env=quadrotor_multi --replay_buffer_sample_prob=0 --quads_use_numba=True \
--train_dir=/home/zhehui/tmp/quad-swarm-rl/swarm_rl/train_dir/hybrid_search_density_slurm/ \
--experiment=00_hybrid_search_density_see_0_q.o.den_0.1 --quads_episode_duration=15.0 \
--quads_view_mode=global --anneal_collision_steps=0 --quads_anneal_safe_start_steps=0 \
--quads_anneal_safe_total_steps=0 --cbf_agg_anneal_steps=0 \
--quads_mode=mix --quads_obst_size=1.0 --quads_obst_gap=0.0 --quads_obst_density=0.2 \
--quads_sbc_only=True --quads_render=False --no_render \
--quads_max_neighbor_aggressive=1.0 --quads_max_obst_aggressive=1.0
