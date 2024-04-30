import ctypes
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
from numpy.ctypeslib import ndpointer

from gym_art.quadrotor_multi.quad_utils import QUADS_NEIGHBOR_OBS_TYPE
from swarm_rl.sim2real.sim2real_utils import load_sf_model


def compare_torch_to_c_model_outputs_single_drone(args):
    # project_root = Path.home().joinpath('quad-swarm-rl')
    # os.chdir(str(project_root))
    # Load a list of torch models and c model names
    models, c_model_names, _ = load_sf_model(Path(args.torch_model_dir), model_type=args.model_type)

    # Get a random observation
    obs = torch.randn((1, 18))
    obs_dict = {'obs': obs}

    for model, c_model_name in zip(models, c_model_names):
        # Get actions from the torch model
        torch_model_out = model.action_parameterization(model.actor_encoder(obs_dict))[1]
        torch_model_out = torch_model_out.means.detach().numpy()

        # Get actions from c model
        c_base_name, c_extension = os.path.splitext(args.output_model_name)
        final_c_model_name = f'{c_base_name}_{c_model_name}{c_extension}'

        c_model_dir = Path(args.output_dir).joinpath(args.model_type)
        c_model_path = Path(args.output_dir).joinpath(args.model_type, final_c_model_name)
        shared_lib_path = c_model_dir.joinpath(f'single_{c_model_name}.so')
        subprocess.run(
            ['g++', '-fPIC', '-shared', '-o', str(shared_lib_path), str(c_model_path)],
            check=True,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE
        )

        lib = ctypes.cdll.LoadLibrary(str(shared_lib_path))
        func = lib.main
        func.restype = None
        func.argtypes = [
            ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
            ctypes.c_size_t,
            ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")
        ]

        indata = obs.flatten().detach().numpy()
        outdata = np.zeros(4).astype(np.float32)
        func(indata, indata.size, outdata)

        assert np.allclose(torch_model_out, outdata, atol=1e-6)


def compare_torch_to_c_model_multi_drone_deepset(args):
    # prepare the c model and main method for evaluation
    parent_model_dir = Path(args.torch_model_dir)
    sub_model_dir_list = [name for name in os.listdir(parent_model_dir) if os.path.isdir(os.path.join(parent_model_dir, name))]
    for i in range(len(sub_model_dir_list)):
        model_dir = parent_model_dir.joinpath(sub_model_dir_list[i])

        models, c_model_names, cfg = load_sf_model(model_dir, model_type=args.model_type)

        for torch_model, c_model_name in zip(models, c_model_names):
            # Get C model
            c_base_name, c_extension = os.path.splitext(args.output_model_name)
            final_c_model_name = f'{c_base_name}_{c_model_name}{c_extension}'
            c_model_dir = Path(args.output_dir).joinpath(args.model_type, model_dir.parts[1], model_dir.parts[2])
            c_model_path = c_model_dir.joinpath(final_c_model_name)
            shared_lib_path = c_model_dir.joinpath(f'multi_deepsets_{c_model_name}.so')

            subprocess.run(
                ['g++', '-fPIC', '-shared', '-o', str(shared_lib_path), str(c_model_path)],
                check=True,
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE
            )

            lib = ctypes.cdll.LoadLibrary(str(shared_lib_path))
            func = lib.main
            func.restype = None
            func.argtypes = [
                ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
            ]

            # test 1000 times on different random inputs
            for test_id in range(1000):
                if test_id % 500 == 0:
                    print(f'Running test {test_id}')
                # Get self obs
                self_obs = torch.randn((1, 18))
                self_indata = self_obs.detach().numpy()

                # Get neighbor obs
                neighbor_obs_size = cfg.quads_neighbor_visible_num * QUADS_NEIGHBOR_OBS_TYPE[cfg.quads_neighbor_obs_type]
                neighbor_obs = torch.randn((1, neighbor_obs_size))
                nbr_indata = neighbor_obs.detach().numpy()

                # Get overall obs
                obs_dict = {'obs': torch.concat([self_obs, neighbor_obs], dim=-1).view(1, -1)}

                # Get torch thrust outputs
                torch_thrust_out = torch_model.action_parameterization(torch_model.actor_encoder(obs_dict))[1]
                torch_thrust_out = torch_thrust_out.means.flatten().detach().numpy()

                # Get C model outputs
                thrust_out = np.zeros(4).astype(np.float32)

                func(self_indata, nbr_indata, thrust_out)

                # print('torch_thrust_out:', torch_thrust_out)
                # print('thrust_out:', thrust_out)

                assert np.allclose(torch_thrust_out, thrust_out, atol=1e-6)


def compare_torch_to_c_model_multi_drone_attention():
    project_root = Path.home().joinpath('quad-swarm-rl')
    os.chdir(str(project_root))

    # prepare the c model and main method for evaluation
    c_model_dir = Path('swarm_rl/sim2real/c_models/attention')
    c_model_path = c_model_dir.joinpath('model.c')
    shared_lib_path = c_model_dir.joinpath('multi_attn.so')
    subprocess.run(
        ['g++', '-fPIC', '-shared', '-o', str(shared_lib_path), str(c_model_path)],
        check=True,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE
    )

    import ctypes
    from numpy.ctypeslib import ndpointer
    lib = ctypes.cdll.LoadLibrary(str(shared_lib_path))
    func = lib.main
    func.restype = None
    func.argtypes = [
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    ]

    torch_model_dir = 'swarm_rl/sim2real/torch_models/attention/'
    model, cfg = load_sf_model(Path(torch_model_dir), model_type='attention')

    # test 1000 times on different random inputs
    for _ in range(1000):
        # check the neighbor encoder outputs
        neighbor_obs = torch.randn(36)
        torch_nbr_out = model.actor_encoder.neighbor_embed_layer(neighbor_obs).detach().numpy()
        nbr_indata = neighbor_obs.detach().numpy()
        nbr_outdata = np.zeros(16).astype(np.float32)

        # check the obstacle encoder outputs
        obstacle_obs = torch.rand(9)
        torch_obstacle_out = model.actor_encoder.obstacle_embed_layer(obstacle_obs).detach().numpy()
        obst_indata = obstacle_obs.detach().numpy()
        obst_outdata = np.zeros(16).astype(np.float32)  # TODO: make this cfg.rnn_size instead of hardcoded

        # check attention layer
        attn_input = torch.from_numpy(np.vstack((torch_nbr_out, torch_obstacle_out)))
        torch_attn_output, _ = model.actor_encoder.attention_layer(attn_input, attn_input, attn_input)
        # torch_attn_output = model.actor_encoder.attention_layer.softmax_out.detach().numpy()
        torch_attn_output = torch_attn_output.detach().numpy()
        token1_out = np.zeros(16).astype(np.float32)
        token2_out = np.zeros(16).astype(np.float32)

        self_obs = torch.randn(19)
        self_indata = self_obs.detach().numpy()
        obs_dict = {'obs': torch.concat([self_obs, neighbor_obs, obstacle_obs]).view(1, -1)}
        torch_thrust_out = model.action_parameterization(model.actor_encoder(obs_dict))[
            1].means.flatten().detach().numpy()
        thrust_out = np.zeros(4).astype(np.float32)

        func(self_indata, nbr_indata, obst_indata, nbr_outdata, obst_outdata, token1_out, token2_out, thrust_out)

        tokens = np.vstack((token1_out, token2_out))
        assert np.allclose(torch_obstacle_out, obst_outdata, atol=1e-6)
        assert np.allclose(torch_nbr_out, nbr_outdata, atol=1e-6)
        assert np.allclose(torch_attn_output, tokens, atol=1e-6)
        assert np.allclose(torch_thrust_out, thrust_out, atol=1e-6)
