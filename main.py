# Training code with logger
# TODO: argparse for running on server
import argparse
import random
import numpy as np
import torch
import os


def wrapper(args):
    pass


def main():
    parser = argparse.ArgumentParser(
        description="RLENA: Battle Arena for RL agents"
    )
    env = ['snake', 'pommerman']
    mode = ['train', 'test', 'demo']

    common = parser.add_argument_group('common configurations')
    common.add_argument("--env", type=str, required=True,
                        choices=env)
    common.add_argument("--mode", type=str, required=True,
                        choices=mode, default='test')
    common.add_argument("--tag", type=str, default='')
    common.add_argument("--seed", type=int, default=-1)

    log = parser.add_argument_group("logging options")
    log.add_argument("--log_level", type=int, default=20)
    log.add_argument("--log_step", type=int, default=100)
    log.add_argument("--save_step", type=int, default=100)
    log.add_argument("--debug", "-d", action="store_true")
    log.add_argument("--quiet", "-q", action="store_true")
    log.add_argument("--maxlen", type=int, default=100)

    dirs = parser.add_argument_group("directory configurations")
    dirs.add_argument("--log_dir", type=str, default='logs')
    dirs.add_argument("--ckpt_dir", type=str, default='ckpt')

    algo = ['DQN', 'RAINBOW', 'PPO', 'QMIX']
    train = parser.add_argument_group("training options")
    train.add_argument("--algo", type=str, required=True,
                       choices=algo)
    train.add_argument("--total_step", type=int, default=100)
    train.add_argument("--env_id", type=str, default=None)
    train.add_argument("--n_env", type=int, default=1)
    train.add_argument("--gpu_id", type=int, default=None)

    model = parser.add_argument_group("Model options")
    model.add_argument("--model", type=str, default=None)
    model.add_argument("--checkpoint", type=str, default=None)

    args = parser.parse_args()

    if args.seed == -1:
        random.seed(None)
        args.seed = random.randrange(0, int(1e4))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.verbose = not args.quiet
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # TODO: GPU allocation with gpu_id
    if args.gpu_id is not None:
        pass
    wrapper(args)


if __name__ == '__main__':
    main()
