import json
import os
from typing import List
from pathlib import Path

import torch
import torch.nn as nn
from attrdict import AttrDict
from sample_factory.model.actor_critic import create_actor_critic

from gym_art.quadrotor_multi.quad_utils import QUADS_NEIGHBOR_OBS_TYPE
from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env_multi
from swarm_rl.sim2real.code_blocks import (
    headers_network_evaluate,
    headers_evaluation,
    headers_multi_deepset_evaluation,
    linear_activation,
    sigmoid_activation,
    relu_activation,
    single_drone_eval,
    multi_drone_attn_eval,
    multi_drone_deepset_eval,
    headers_multi_agent_attention,
    attention_body
)
from swarm_rl.train import register_swarm_components


def load_sf_model(model_dir, model_type):
    """
        Load a trained SF pytorch model
    """
    assert model_dir.exists(), f'Path {str(model_dir)} is not a valid path'
    # Load hyper-parameters
    cfg_path = model_dir.joinpath('config.json')
    # Compatibility with sf2
    if not cfg_path.exists():
        cfg_path = model_dir.joinpath('cfg.json')

    with open(cfg_path, 'r') as f:
        args = json.load(f)
    args = AttrDict(args)

    # Manually set some values
    args.visualize_v_value = False
    args.quads_encoder_type = 'attention' if model_type == 'attention' else 'corl'
    args.quads_sim2real = True
    args.quads_domain_random = False
    args.quads_obst_density_random = False
    args.quads_obst_density_min = 0
    args.quads_obst_density_max = 0
    args.quads_obst_size_random = False
    args.quads_obst_size_min = 0
    args.quads_obst_size_max = 0
    args.quads_obs_rel_rot = False
    args.quads_obst_noise = 0.0
    args.quads_obst_grid_size = 1.0
    args.quads_render_mode = 'human'

    # Load model
    register_swarm_components()
    # Spawn a dummy env, so we can get the obs and action space info
    env = make_quadrotor_env_multi(args)
    # Get checkpoints under checkpoint_p0
    checkpoint_dir = Path(os.path.join(model_dir, 'checkpoint_p0'))
    model_paths_1 = list(checkpoint_dir.glob('*.pth'))

    # Get checkpoints under checkpoint_p0/milestones
    milestone_dir = Path(os.path.join(model_dir, 'checkpoint_p0/milestones'))
    model_paths_2 = list(milestone_dir.glob('*.pth'))

    model_paths = model_paths_1 + model_paths_2

    models = []
    c_model_names = []
    for model_path in model_paths:
        model = create_actor_critic(args, env.observation_space, env.action_space)
        model.load_state_dict(torch.load(model_path)['model'])
        models.append(model)

        # Extract the step number from the model path
        split_path = model_path.stem.split('_')
        prex_name = split_path[0]
        train_step = split_path[-1]

        if 'best' in prex_name:
            c_model_name = f'best_step_{train_step}'
        else:
            c_model_name = f'step_{train_step}'

        c_model_names.append(c_model_name)

    return models, c_model_names, args


def process_layer(name, param, layer_type):
    """
    Convert a torch parameter from the NN into a c-equivalent represented as a string
    """
    if layer_type == 'weight':
        weight = 'static const float ' + name + '[' + str(param.shape[0]) + '][' + str(param.shape[1]) + '] = {'
        for row in param:
            weight += '{'
            for num in row:
                weight += str(num.item()) + ','
            # get rid of comma after the last number
            weight = weight[:-1]
            weight += '},'
        # get rid of comma after the last curly bracket
        weight = weight[:-1]
        weight += '};\n'
        return weight
    else:
        bias = 'static const float ' + name + '[' + str(param.shape[0]) + '] = {'
        for num in param:
            bias += str(num.item()) + ','
        # get rid of comma after last number
        bias = bias[:-1]
        bias += '};\n'
        return bias


def generate_c_weights_attention(model: nn.Module, transpose: bool = False):
    """
            Generate c friendly weight strings for the c version of the attention model
            order is: self-encoder, neighbor-encoder, obst-encoder, attention, then final combined output layers
    """
    self_weights, self_biases, self_layer_names, self_bias_names = [], [], [], []
    neighbor_weights, neighbor_biases, nbr_layer_names, nbr_bias_names = [], [], [], []
    obst_weights, obst_biases, obst_layer_names, obst_bias_names = [], [], [], []
    attn_weights, attn_biases, attn_layer_names, attn_bias_names = [], [], [], []
    out_weights, out_biases, out_layer_names, out_bias_names = [], [], [], [],
    outputs = []
    n_self, n_nbr, n_obst = 0, 0, 0
    for name, param in model.named_parameters():
        # get the self encoder weights
        if transpose:
            param = param.T
        c_name = name.replace('.', '_')
        if 'weight' in c_name and 'critic' not in c_name and 'layer_norm' not in c_name:
            weight = process_layer(c_name, param, layer_type='weight')
            if 'self_embed' in c_name:
                self_layer_names.append(name)
                self_weights.append(weight)
                outputs.append('static float output_' + str(n_self) + '[' + str(param.shape[1]) + '];\n')
                n_self += 1
            elif 'neighbor_embed' in c_name:
                nbr_layer_names.append(name)
                neighbor_weights.append(weight)
                outputs.append('static float nbr_output_' + str(n_nbr) + '[' + str(param.shape[1]) + '];\n')
                n_nbr += 1
            elif 'obstacle_embed' in c_name:
                obst_layer_names.append(name)
                obst_weights.append(weight)
                outputs.append('static float obst_output_' + str(n_obst) + '[' + str(param.shape[1]) + '];\n')
                n_obst += 1
            elif 'attention' in c_name or 'layer_norm' in c_name:
                attn_layer_names.append(name)
                attn_weights.append(weight)
            else:
                # output layer
                out_layer_names.append(name)
                out_weights.append(weight)
                # these will be considered part of the self encoder
                outputs.append('static float output_' + str(n_self) + '[' + str(param.shape[1]) + '];\n')
                n_self += 1
        if ('bias' in c_name or 'layer_norm' in c_name) and 'critic' not in c_name:
            bias = process_layer(c_name, param, layer_type='bias')
            if 'self_embed' in c_name:
                self_bias_names.append(name)
                self_biases.append(bias)
            elif 'neighbor_embed' in c_name:
                nbr_bias_names.append(name)
                neighbor_biases.append(bias)
            elif 'obstacle_embed' in c_name:
                obst_bias_names.append(name)
                obst_biases.append(bias)
            elif 'attention' in c_name or 'layer_norm' in c_name:
                attn_bias_names.append(name)
                attn_biases.append(bias)
            else:
                # output layer
                out_bias_names.append(name)
                out_biases.append(bias)

    self_layer_names += out_layer_names
    self_bias_names += out_bias_names
    self_weights += out_weights
    self_biases += out_biases
    info = {
        'encoders': {
            'self': [self_layer_names, self_bias_names, self_weights, self_biases],
            'nbr': [nbr_layer_names, nbr_bias_names, neighbor_weights, neighbor_biases],
            'obst': [obst_layer_names, obst_bias_names, obst_weights, obst_biases],
            'attn': [attn_layer_names, attn_bias_names, attn_weights, attn_biases],
        },
        'out': [out_layer_names, out_bias_names, out_weights, out_biases],
        'outputs': outputs
    }

    return info


def generate_c_weights_multi_deepset(model: nn.Module, transpose: bool = False):
    """
        Generate c friendly weight strings for the c version of the multi-agent deepset model
        The order is self-encoder, neighbor-encoder and final combined output layers

        Model architecture can be found in swarm_rl/quad_multi_model.py
    """
    self_weights, self_biases, self_layer_names, self_bias_names = [], [], [], []
    neighbor_weights, neighbor_biases, nbr_layer_names, nbr_bias_names = [], [], [], []
    out_weights, out_biases, out_layer_names, out_bias_names = [], [], [], [],
    outputs = []
    n_self, n_nbr = 0, 0

    for name, param in model.named_parameters():
        # first get the self encoder weights
        if transpose:
            param = param.T
        c_name = name.replace('.', '_')
        if 'weight' in c_name and 'critic' not in c_name and 'layer_norm' not in c_name:
            weight = process_layer(c_name, param, layer_type='weight')
            if 'self_embed' in c_name:
                self_layer_names.append(name)
                self_weights.append(weight)
                outputs.append('static float output_' + str(n_self) + '[' + str(param.shape[1]) + '];\n')
                n_self += 1
            elif 'neighbor_encoder' in c_name:
                nbr_layer_names.append(name)
                neighbor_weights.append(weight)
                outputs.append(
                    'static float nbr_output_' + str(n_nbr) + '[NEIGHBORS]' + '[' + str(param.shape[1]) + '];\n')
                n_nbr += 1
            else:
                # output layer
                out_layer_names.append(name)
                out_weights.append(weight)
                # these will be considered part of the self encoder
                outputs.append('static float output_' + str(n_self) + '[' + str(param.shape[1]) + '];\n')
                n_self += 1

        if ('bias' in c_name or 'layer_norm' in c_name) and 'critic' not in c_name:
            bias = process_layer(c_name, param, layer_type='bias')
            if 'self_embed' in c_name:
                self_bias_names.append(name)
                self_biases.append(bias)
            elif 'neighbor_encoder' in c_name:
                nbr_bias_names.append(name)
                neighbor_biases.append(bias)
            else:
                # output layer
                out_bias_names.append(name)
                out_biases.append(bias)

    self_layer_names += out_layer_names
    self_bias_names += out_bias_names
    self_weights += out_weights
    self_biases += out_biases
    info = {
        'encoders': {
            'self': [self_layer_names, self_bias_names, self_weights, self_biases],
            'nbr': [nbr_layer_names, nbr_bias_names, neighbor_weights, neighbor_biases],
        },
        'out': [out_layer_names, out_bias_names, out_weights, out_biases],
        'outputs': outputs
    }

    return info


def generate_c_weights(model: nn.Module, transpose: bool = False):
    """
        Generate c friendly weight strings for the c version of the single drone model
    """
    weights, biases = [], []
    layer_names, bias_names, outputs = [], [], []
    n_bias = 0
    for name, param in model.named_parameters():
        if transpose:
            param = param.T
        name = name.replace('.', '_')
        if 'weight' in name and 'critic' not in name and 'layer_norm' not in name:
            layer_names.append(name)
            weight = 'static const float ' + name + '[' + str(param.shape[0]) + '][' + str(param.shape[1]) + '] = {'
            for row in param:
                weight += '{'
                for num in row:
                    weight += str(num.item()) + ','
                # get rid of comma after the last number
                weight = weight[:-1]
                weight += '},'
            # get rid of comma after the last curly bracket
            weight = weight[:-1]
            weight += '};\n'
            weights.append(weight)

        if 'bias' in name and 'critic' not in name:
            bias_names.append(name)
            bias = 'static const float ' + name + '[' + str(param.shape[0]) + '] = {'
            for num in param:
                bias += str(num.item()) + ','
            # get rid of comma after last number
            bias = bias[:-1]
            bias += '};\n'
            biases.append(bias)
            output = 'static float output_' + str(n_bias) + '[' + str(param.shape[0]) + '];\n'
            outputs.append(output)
            n_bias += 1

    return layer_names, bias_names, weights, biases, outputs


def self_encoder_c_str(prefix: str, weight_names: List[str], bias_names: List[str]):
    method = """\nvoid networkEvaluate(struct control_t_n *control_n, const float *state_array) {"""
    num_layers = len(weight_names)
    # write the for loops for forward-prop
    for_loops = []
    input_for_loop = f'''
    for (int i = 0; i < {prefix}_structure[0][1]; i++) {{
        output_0[i] = 0;
        for (int j = 0; j < {prefix}_structure[0][0]; j++) {{
            output_0[i] += state_array[j] * {weight_names[0].replace('.', '_')}[j][i];
        }}
        output_0[i] += {bias_names[0].replace('.', '_')}[i];
        output_0[i] = tanhf(output_0[i]);
    }}
'''
    for_loops.append(input_for_loop)

    # rest of the hidden layers
    for n in range(1, num_layers - 2):
        for_loop = f'''
    for (int i = 0; i < {prefix}_structure[{str(n)}][1]; i++) {{
        output_{str(n)}[i] = 0;
        for (int j = 0; j < {prefix}_structure[{str(n)}][0]; j++) {{
            output_{str(n)}[i] += output_{str(n - 1)}[j] * {weight_names[n].replace('.', '_')}[j][i];
        }}
        output_{str(n)}[i] += {bias_names[n].replace('.', '_')}[i];
        output_{str(n)}[i] = tanhf(output_{str(n)}[i]);
    }}
'''
        for_loops.append(for_loop)

    # Concat self embedding and neighbor embedding
    for_loop = f'''
    // Concat self_embed and neighbor_embed
    for (int i = 0; i < D_SELF; i++) {{
        output_embeds[i] = output_{num_layers - 3}[i];
    }}
    for (int i = 0; i < D_NBR; i++) {{
        output_embeds[D_SELF + i] = neighbor_embeds[i];
    }}
'''
    for_loops.append(for_loop)

    # forward-prop of feedforward layer
    output_for_loop = f'''
    // Feedforward layer
    for (int i = 0; i < self_structure[2][1]; i++) {{
        output_{num_layers - 2}[i] = 0;
        for (int j = 0; j < self_structure[2][0] ; j++) {{
            output_{num_layers - 2}[i] += output_embeds[j] * actor_encoder_feed_forward_0_weight[j][i];
        }}
        output_{num_layers - 2}[i] += actor_encoder_feed_forward_0_bias[i];
        output_{num_layers - 2}[i] = tanhf(output_2[i]);
    }}
'''
    for_loops.append(output_for_loop)

    # the last hidden layer which is supposed to have no non-linearity
    n = num_layers - 1
    output_for_loop = f'''
    for (int i = 0; i < {prefix}_structure[{str(n)}][1]; i++) {{
        output_{str(n)}[i] = 0;
        for (int j = 0; j < {prefix}_structure[{str(n)}][0]; j++) {{
            output_{str(n)}[i] += output_{str(n - 1)}[j] * {weight_names[n].replace('.', '_')}[j][i];
        }}
        output_{str(n)}[i] += {bias_names[n].replace('.', '_')}[i];
    }}
'''
    for_loops.append(output_for_loop)

    for code in for_loops:
        method += code
    if 'self' in prefix:
        # assign network outputs to control
        assignment = """
    control_n->thrust_0 = output_""" + str(n) + """[0];
    control_n->thrust_1 = output_""" + str(n) + """[1];
    control_n->thrust_2 = output_""" + str(n) + """[2];
    control_n->thrust_3 = output_""" + str(n) + """[3];
"""
        method += assignment
    # closing bracket
    method += """}\n\n"""
    return method


def self_encoder_attn_c_str(prefix: str, weight_names: List[str], bias_names: List[str]):
    method = """void networkEvaluate(struct control_t_n *control_n, const float *state_array) {"""
    num_layers = len(weight_names)
    # write the for loops for forward-prop of self embed layer
    for_loops = []
    input_for_loop = f'''
        // Self embed layer
        for (int i = 0; i < {prefix}_structure[0][1]; i++) {{
            output_0[i] = 0;
            for (int j = 0; j < {prefix}_structure[0][0]; j++) {{
                output_0[i] += state_array[j] * {weight_names[0].replace('.', '_')}[j][i];
            }}
            output_0[i] += {bias_names[0].replace('.', '_')}[i];
            output_0[i] = tanhf(output_0[i]);
        }}
    '''
    for_loops.append(input_for_loop)

    # concat self embedding and attention embedding
    # for n in range(1, num_layers - 1):
    for_loop = f'''
        // Concat self_embed, neighbor_embed and obst_embed
        for (int i = 0; i < self_structure[0][1]; i++) {{
            output_embeds[i] = output_0[i];
            output_embeds[i + self_structure[0][1]] = attn_embeds[0][i];
            output_embeds[i + 2 * self_structure[0][1]] = attn_embeds[1][i];
        }}
    '''
    for_loops.append(for_loop)

    # forward-prop of feedforward layer
    output_for_loop = f'''
        // Feedforward layer
        for (int i = 0; i < self_structure[1][1]; i++) {{
            output_1[i] = 0;
            for (int j = 0; j < 3 * self_structure[0][1]; j++) {{
                output_1[i] += output_embeds[j] * actor_encoder_feed_forward_0_weight[j][i];
                }}
            output_1[i] += actor_encoder_feed_forward_0_bias[i];
            output_1[i] = tanhf(output_1[i]);
        }}
    '''
    for_loops.append(output_for_loop)

    # the last hidden layer which is supposed to have no non-linearity
    n = num_layers - 1
    output_for_loop = f'''
        for (int i = 0; i < {prefix}_structure[{str(n)}][1]; i++) {{
            output_{str(n)}[i] = 0;
            for (int j = 0; j < {prefix}_structure[{str(n)}][0]; j++) {{
                output_{str(n)}[i] += output_{str(n - 1)}[j] * {weight_names[n].replace('.', '_')}[j][i];
            }}
            output_{str(n)}[i] += {bias_names[n].replace('.', '_')}[i];
        }}
        '''
    for_loops.append(output_for_loop)

    for code in for_loops:
        method += code
    if 'self' in prefix:
        # assign network outputs to control
        assignment = """
        control_n->thrust_0 = output_""" + str(n) + """[0];
        control_n->thrust_1 = output_""" + str(n) + """[1];
        control_n->thrust_2 = output_""" + str(n) + """[2];
        control_n->thrust_3 = output_""" + str(n) + """[3];	
    """
        method += assignment
    # closing bracket
    method += """}\n\n"""
    return method


def neighbor_encoder_deepset_c_string(prefix: str, weight_names: List[str], bias_names: List[str]):
    method = """void neighborEmbedder(const float neighbor_inputs[NEIGHBORS*NBR_OBS_DIM]) {"""
    num_layers = len(weight_names)

    # write the for loops for forward-prop
    for_loops = []
    input_for_loop = f'''
    for (int n = 0; n < NEIGHBORS; n++) {{            
        for (int i = 0; i < {prefix}_structure[0][1]; i++) {{
            {prefix}_output_0[n][i] = 0; 
            for (int j = 0; j < {prefix}_structure[0][0]; j++) {{
                {prefix}_output_0[n][i] += neighbor_inputs[n*NBR_OBS_DIM + j] * actor_encoder_neighbor_encoder_embedding_mlp_0_weight[j][i]; 
            }}
            {prefix}_output_0[n][i] += actor_encoder_neighbor_encoder_embedding_mlp_0_bias[i];
            {prefix}_output_0[n][i] = tanhf({prefix}_output_0[n][i]);
        }}
    }}
'''
    for_loops.append(input_for_loop)

    # hidden layers
    for n in range(1, num_layers):
        for_loop = f'''
    for (int n = 0; n < NEIGHBORS; n++) {{
        for (int i = 0; i < {prefix}_structure[{str(n)}][1]; i++) {{
            {prefix}_output_{str(n)}[n][i] = 0;
            for (int j = 0; j < {prefix}_structure[{str(n)}][0]; j++) {{
                {prefix}_output_{str(n)}[n][i] += {prefix}_output_{str(n - 1)}[n][j] * {weight_names[n].replace('.', '_')}[j][i];
            }}
            {prefix}_output_{str(n)}[n][i] += {bias_names[n].replace('.', '_')}[i];
            {prefix}_output_{str(n)}[n][i] = tanhf({prefix}_output_{str(n)}[n][i]);
        }}
    }}
'''
        for_loops.append(for_loop)

    # Average the neighbor embeddings
    for_loop = f'''
    // Average over number of neighbors
    for (int i = 0; i < D_NBR; i++) {{
        neighbor_embeds[i] = 0;
        for (int n = 0; n < NEIGHBORS; n++) {{
            neighbor_embeds[i] += {prefix}_output_{str(num_layers - 1)}[n][i];
        }}
        neighbor_embeds[i] /= NEIGHBORS;
    }}
'''
    for_loops.append(for_loop)

    for code in for_loops:
        method += code
    # method closing bracket
    method += """}\n"""
    return method


def neighbor_encoder_c_string(prefix: str, weight_names: List[str], bias_names: List[str]):
    method = """void neighborEmbedder(const float neighbor_inputs[NEIGHBORS * NBR_DIM]) {
    """
    num_layers = len(weight_names)

    # write the for loops for forward-prop
    for_loops = []
    input_for_loop = f'''
            for (int i = 0; i < {prefix}_structure[0][1]; i++) {{
                {prefix}_output_0[i] = 0; 
                for (int j = 0; j < {prefix}_structure[0][0]; j++) {{
                    {prefix}_output_0[i] += neighbor_inputs[j] * actor_encoder_neighbor_embed_layer_0_weight[j][i]; 
                }}
                {prefix}_output_0[i] += actor_encoder_neighbor_embed_layer_0_bias[i];
                {prefix}_output_0[i] = tanhf({prefix}_output_0[i]);
            }}
    '''
    for_loops.append(input_for_loop)

    # hidden layers
    for n in range(1, num_layers - 1):
        for_loop = f'''
                for (int i = 0; i < {prefix}_structure[{str(n)}][1]; i++) {{
                    {prefix}_output_{str(n)}[i] = 0;
                    for (int j = 0; j < {prefix}_structure[{str(n)}][0]; j++) {{
                        output_{str(n)}[i] += output_{str(n - 1)}[j] * {weight_names[n].replace('.', '_')}[j][i];
                    }}
                    output_{str(n)}[i] += {bias_names[n].replace('.', '_')}[i];
                    output_{str(n)}[i] = tanhf(output_{str(n)}[i]);
                }}
        '''
        for_loops.append(for_loop)

    # the last hidden layer which is supposed to have no non-linearity
    n = num_layers - 1
    if n > 0:
        output_for_loop = f'''
                for (int i = 0; i < {prefix}_structure[{str(n)}][1]; i++) {{
                    output_{str(n)}[i] = 0;
                    for (int j = 0; j < {prefix}_structure[{str(n)}][0]; j++) {{
                        output_{str(n)}[i] += output_{str(n - 1)}[j] * {weight_names[n].replace('.', '_')}[j][i];
                    }}
                    output_{str(n)}[i] += {bias_names[n].replace('.', '_')}[i];
                    neighbor_embeds[i] += output_{str(n)}[i]; 
                }}
        '''
        for_loops.append(output_for_loop)

    for code in for_loops:
        method += code
    # method closing bracket
    method += """}\n\n"""
    return method


def obstacle_encoder_c_str(prefix: str, weight_names: List[str], bias_names: List[str]):
    method = f"""void obstacleEmbedder(const float obstacle_inputs[OBST_DIM]) {{
        //reset embeddings accumulator to zero
        memset(obstacle_embeds, 0, sizeof(obstacle_embeds));

    """
    num_layers = len(weight_names)

    # write the for loops for forward-prop
    for_loops = []
    input_for_loop = f'''
        for (int i = 0; i < {prefix}_structure[0][1]; i++) {{
            {prefix}_output_0[i] = 0;
            for (int j = 0; j < {prefix}_structure[0][0]; j++) {{
                {prefix}_output_0[i] += obstacle_inputs[j] * {weight_names[0].replace('.', '_')}[j][i];
            }}
            {prefix}_output_0[i] += {bias_names[0].replace('.', '_')}[i];
            {prefix}_output_0[i] = tanhf({prefix}_output_0[i]);
        }}
    '''
    for_loops.append(input_for_loop)

    # rest of the hidden layers
    for n in range(1, num_layers - 1):
        for_loop = f'''
            for (int i = 0; i < {prefix}_structure[{str(n)}][1]; i++) {{
                output_{str(n)}[i] = 0;
                for (int j = 0; j < {prefix}_structure[{str(n)}][0]; j++) {{
                    output_{str(n)}[i] += output_{str(n - 1)}[j] * {weight_names[n].replace('.', '_')}[j][i];
                }}
                output_{str(n)}[i] += {bias_names[n].replace('.', '_')}[i];
                output_{str(n)}[i] = tanhf(output_{str(n)}[i]);
            }}
        '''
        for_loops.append(for_loop)

    # the last hidden layer which is supposed to have no non-linearity
    n = num_layers - 1
    if n > 0:
        output_for_loop = f'''
            for (int i = 0; i < {prefix}_structure[{str(n)}][1]; i++) {{
                output_{str(n)}[i] = 0;
                for (int j = 0; j < {prefix}_structure[{str(n)}][0]; j++) {{
                    output_{str(n)}[i] += output_{str(n - 1)}[j] * {weight_names[n].replace('.', '_')}[j][i];
                }}
                output_{str(n)}[i] += {bias_names[n].replace('.', '_')}[i];
                obstacle_embeds[i] += output_{str(n)}[i];
            }}
        '''
        for_loops.append(output_for_loop)

    for code in for_loops:
        method += code
    # closing bracket
    method += """}\n\n"""
    return method


def generate_c_model_attention(model: nn.Module, output_path: str, output_folder: str, testing=False):
    info = generate_c_weights_attention(model, transpose=True)
    model_state_dict = model.state_dict()

    source = ""
    structures = ""
    methods = ""

    # setup all the encoders
    for enc_name, data in info['encoders'].items():
        # data contains [weight_names, bias_names, weights, biases]
        structure = f'static const int {enc_name}_structure [' + str(int(len(data[0]))) + '][2] = {'

        weight_names, bias_names = data[0], data[1]
        for w_name, b_name in zip(weight_names, bias_names):
            w = model_state_dict[w_name].T
            structure += '{' + str(w.shape[0]) + ', ' + str(w.shape[1]) + '},'

        # complete the structure array
        # get rid of the comma after the last curly bracket
        structure = structure[:-1]
        structure += '};\n'
        structures += structure

        if 'self' in enc_name:
            method = self_encoder_attn_c_str(enc_name, weight_names, bias_names)
        elif 'nbr' in enc_name:
            method = neighbor_encoder_c_string(enc_name, weight_names, bias_names)
        elif 'obst' in enc_name:
            method = obstacle_encoder_c_str(enc_name, weight_names, bias_names)
        else:
            # attention
            method = attention_body

        methods += method

    # headers
    source += headers_network_evaluate if not testing else headers_evaluation
    source += headers_multi_agent_attention

    # helper funcs
    source += linear_activation
    source += sigmoid_activation
    source += relu_activation

    # network eval func
    source += structures
    outputs = info['outputs']
    for output in outputs:
        source += output

    encoders = info['encoders']

    for key, vals in encoders.items():
        weights, biases = vals[-2], vals[-1]
        for w in weights:
            source += w
        for b in biases:
            source += b

    source += methods

    if testing:
        source += multi_drone_attn_eval

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(source)
        f.close()

    return source


def generate_c_model_multi_deepset(model: nn.Module, output_path: str, output_folder: str, testing=False, cfg=None):
    """
        Generate c model for the multi-agent deepset model
    """
    model_state_dict = model.state_dict()
    # for name, param in model_state_dict.items():
    #     print(name, param.shape)

    infos = generate_c_weights_multi_deepset(model, transpose=True)
    model_state_dict = model.state_dict()

    source = ""
    structures = ""
    methods = ""

    # setup all the encoders
    for enc_name, data in infos['encoders'].items():
        # data contains [weight_names, bias_names, weights, biases]
        structure = f'static const int {enc_name}_structure [' + str(int(len(data[0]))) + '][2] = {'

        weight_names, bias_names = data[0], data[1]
        for w_name, b_name in zip(weight_names, bias_names):
            w = model_state_dict[w_name].T
            structure += '{' + str(w.shape[0]) + ', ' + str(w.shape[1]) + '},'

        # complete the structure array
        # get rid of the comma after the last curly bracket
        structure = structure[:-1]
        structure += '};\n'
        structures += structure

        method = ""
        if 'self' in enc_name:
            method = self_encoder_c_str(enc_name, weight_names, bias_names)
        elif 'nbr' in enc_name:
            method = neighbor_encoder_deepset_c_string(enc_name, weight_names, bias_names)
        methods += method

    structures += """
static const int D_SELF = self_structure[1][1];
static const int D_NBR = nbr_structure[1][1];

static float neighbor_embeds[D_NBR];
static float output_embeds[D_SELF + D_NBR];

"""

    # Source code: headers + helper funcs + structures + methods
    # headers
    source += headers_network_evaluate if not testing else headers_multi_deepset_evaluation

    if testing:
        neighbor_obs_info = f"""static const int NEIGHBORS = {cfg.quads_neighbor_visible_num};
static const int NBR_OBS_DIM = {QUADS_NEIGHBOR_OBS_TYPE[cfg.quads_neighbor_obs_type]};

"""
        source += neighbor_obs_info

    # helper funcs
    # source += linear_activation
    # source += sigmoid_activation
    # source += relu_activation

    # network eval func
    source += structures
    outputs = infos['outputs']
    for output in outputs:
        source += output

    encoders = infos['encoders']

    for key, vals in encoders.items():
        weights, biases = vals[-2], vals[-1]
        for w in weights:
            source += w
        for b in biases:
            source += b

    source += methods

    if testing:
        source += multi_drone_deepset_eval

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(source)
        f.close()

    return source


def generate_c_model(model: nn.Module, output_path: str, output_folder: str, testing=False):
    layer_names, bias_names, weights, biases, outputs = generate_c_weights(model, transpose=True)
    num_layers = len(layer_names)

    structure = 'static const int structure [' + str(int(num_layers)) + '][2] = {'
    for name, param in model.named_parameters():
        param = param.T
        if 'weight' in name and 'critic' not in name and 'layer_norm' not in name:
            structure += '{' + str(param.shape[0]) + ', ' + str(param.shape[1]) + '},'

    # complete the structure array
    # get rid of the comma after the last curly bracket
    structure = structure[:-1]
    structure += '};\n'

    # write the for loops for forward-prop
    for_loops = []
    input_for_loop = f'''
    for (int i = 0; i < structure[0][1]; i++) {{
        output_0[i] = 0;
        for (int j = 0; j < structure[0][0]; j++) {{
            output_0[i] += state_array[j] * {layer_names[0]}[j][i];
        }}
        output_0[i] += {bias_names[0]}[i];
        output_0[i] = tanhf(output_0[i]);
    }}
    '''
    for_loops.append(input_for_loop)

    # rest of the hidden layers
    for n in range(1, num_layers - 1):
        for_loop = f'''
    for (int i = 0; i < structure[{str(n)}][1]; i++) {{
        output_{str(n)}[i] = 0;
        for (int j = 0; j < structure[{str(n)}][0]; j++) {{
            output_{str(n)}[i] += output_{str(n - 1)}[j] * {layer_names[n]}[j][i];
        }}
        output_{str(n)}[i] += {bias_names[n]}[i];
        output_{str(n)}[i] = tanhf(output_{str(n)}[i]);
    }}
    '''
        for_loops.append(for_loop)

    # the last hidden layer which is supposed to have no non-linearity
    n = num_layers - 1
    output_for_loop = f'''
    for (int i = 0; i < structure[{str(n)}][1]; i++) {{
        output_{str(n)}[i] = 0;
        for (int j = 0; j < structure[{str(n)}][0]; j++) {{
            output_{str(n)}[i] += output_{str(n - 1)}[j] * {layer_names[n]}[j][i];
        }}
        output_{str(n)}[i] += {bias_names[n]}[i];
    }}
    '''
    for_loops.append(output_for_loop)

    # assign network outputs to control
    assignment = """
    control_n->thrust_0 = output_""" + str(n) + """[0];
    control_n->thrust_1 = output_""" + str(n) + """[1];
    control_n->thrust_2 = output_""" + str(n) + """[2];
    control_n->thrust_3 = output_""" + str(n) + """[3];
"""

    # construct the network evaluate function
    controller_eval = """\nvoid networkEvaluate(struct control_t_n *control_n, const float *state_array) {"""
    for code in for_loops:
        controller_eval += code
    # assignment to control_n
    controller_eval += assignment

    # closing bracket
    controller_eval += """}"""

    # combine all the codes
    source = ""
    # headers
    source += headers_network_evaluate if not testing else headers_evaluation
    # helper funcs
    # source += linear_activation
    # source += sigmoid_activation
    # source += relu_activation
    # network eval func
    source += structure
    for output in outputs:
        source += output
    for weight in weights:
        source += weight
    for bias in biases:
        source += bias
    source += controller_eval

    if testing:
        source += single_drone_eval

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(source)
        f.close()

    return source
