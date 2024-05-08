import argparse
import sys
from attrdict import AttrDict

from swarm_rl.sim2real.tests.unit_test_utils import (compare_torch_to_c_model_outputs_single_drone,
                                                     compare_torch_to_c_model_multi_drone_deepset,
                                                     compare_torch_to_c_model_multi_drone_attention,
                                                     compare_torch_to_c_model_outputs_single_obst)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch_model_dir', type=str,
                        default='torch_models/single',
                        help='Path where the policy and config file is stored')
    parser.add_argument('--output_dir', type=str,
                        default='c_models',
                        help='Where you want the c model to be saved')
    parser.add_argument('--output_model_name', type=str,
                        default='test.c',
                        help='Name of the generated c model file')
    parser.add_argument('--model_type', type=str,
                        default='single',
                        choices=['single', 'single_obst', 'multi_deepset', 'multi_obst_attention'],
                        help='What kind of model we are working with. '
                             'single: single drone, without neighbor encoder, without obstacle encoder.'
                             'single_obst: single drone, without neighbor encoder, with obstacle encoder.'
                             'multi_deepset: multiple drones, neighbor encoder: deepset, obstacle encoder: N/A.'
                             'multi_obst_attention: multiple drones, neighbor encoder: attention, obstacle encoder: N/A'
                        )

    args = parser.parse_args()
    return AttrDict(vars(args))


def main():
    args = parse_args()
    if args.model_type == 'single':
        compare_torch_to_c_model_outputs_single_drone(args=args)
    elif args.model_type == 'single_obst':
        compare_torch_to_c_model_outputs_single_obst(args=args)
    elif args.model_type == 'multi_deepset':
        compare_torch_to_c_model_multi_drone_deepset(args=args)
    elif args.model_type == 'multi_obst_attention':
        compare_torch_to_c_model_multi_drone_attention()
    else:
        raise NotImplementedError(f'Model type {args.model_type} is not supported')

    print('Pass Unit Test!')


if __name__ == '__main__':
    sys.exit(main())
