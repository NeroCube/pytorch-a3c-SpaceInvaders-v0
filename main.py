import argparse
import os
import sys

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

from environment import create_atari_env


parser = argparse.ArgumentParser(description='A3C')

# 參數部分
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--tau', type=float, default=1.00, metavar='T',
                    help='parameter for GAE (default: 1.00)')
# 演算法部分
parser.add_argument('--num-processes', type=int, default=4, metavar='NP',
                    help='how many training processes to use (default: 4)')
parser.add_argument('--num-steps', type=int, default=20, metavar='NS',
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--max-episode-length', type=int, default=100000, metavar='M',
                    help='maximun length of an episode (default: 100000)')
parser.add_argument('--env-name', default='SpaceInvaders-v0', metavar='ENV',
                    help='environment to train on (default: Pong-v0)')

if __name__ == '__main__':
    # 控制 Thread 的數量
    os.environ['OMP_NUM_THREADS'] = '1'
    args = parser.parse_args()
    env = create_atari_env(args.env_name)

    