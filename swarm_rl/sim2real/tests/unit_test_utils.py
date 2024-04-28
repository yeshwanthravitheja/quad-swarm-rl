import ctypes
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
from numpy.ctypeslib import ndpointer

from swarm_rl.sim2real.sim2real_utils import load_sf_model


def compare_torch_to_c_model_outputs_single_drone(args):
    # project_root = Path.home().joinpath('quad-swarm-rl')
    # os.chdir(str(project_root))
    # Load a list of torch models and c model names
    models, c_model_names = load_sf_model(Path(args.torch_model_dir), model_type=args.model_type)

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

        assert np.allclose(torch_model_out, outdata)


def compare_torch_to_c_model_multi_drone_deepset():
    project_root = Path.home().joinpath('quad-swarm-rl')
    os.chdir(str(project_root))

    # prepare the c model and main method for evaluation
    c_model_dir = Path('swarm_rl/sim2real/c_models/multi_deepset')
    c_model_path = c_model_dir.joinpath('model.c')
    shared_lib_path = c_model_dir.joinpath('multi_deepset.so')
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
    ]

    torch_model_dir = 'swarm_rl/sim2real/torch_models/multi_deepset'
    torch_model, cfg = load_sf_model(Path(torch_model_dir), model_type='corl')

    # test 1000 times on different random inputs
    for _ in range(1000):
        self_obs = torch.randn((1, 18))
        self_indata = self_obs.detach().numpy()
        thrust_out = np.zeros(4).astype(np.float32)

        neighbor_obs = torch.randn((1, cfg.quads_neighbor_visible_num * 6))
        nbr_indata = neighbor_obs.detach().numpy()

        obs_dict = {'obs': torch.concat([self_obs, neighbor_obs], dim=-1).view(1, -1)}
        torch_thrust_out = torch_model.action_parameterization(torch_model.actor_encoder(obs_dict))[
            1].means.flatten().detach().numpy()

        func(self_indata, nbr_indata, thrust_out)

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
