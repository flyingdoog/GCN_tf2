import argparse
import numpy as np
import os
import random


def get_params():
    ''' Get parameters from command line '''
    parser = argparse.ArgumentParser()

    # settings
    parser.add_argument("--dataset", type=str, default='cora', help="Dataset string")# 'cora', 'citeseer', 'pubmed'
    parser.add_argument('--id', type=str, default='default_id', help='id to store in database')  #
    parser.add_argument('--device', type=int, default=0,help='device to use')  #
    parser.add_argument('--setting', type=str, default="description of hyper-parameters.")  #
    parser.add_argument('--task_type', type=str, default='semi')
    parser.add_argument('--early_stop', type=int, default= 30, help='early_stop')
    parser.add_argument('--dtype', type=str, default='float32')  #
    parser.add_argument('--seed',type=int, default=1234, help='seed')
    parser.add_argument('--record',type=bool, default=False, help='write to database, for tuning.')
    parser.add_argument('--machine', type=str, default='local')  #

    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
    parser.add_argument('--dropout',type=float, default=0.1, help='dropout rate (1 - keep probability).')
    parser.add_argument('--weight_decay',type=float, default=5e-4, help='Weight for L2 loss on embedding matrix.')
    parser.add_argument('--hiddens', type=str, default='256')
    parser.add_argument("--lr", type=float, default=0.01,help='initial learning rate.')
    parser.add_argument('--act', type=str, default='relu', help='activation funciton')  #
    parser.add_argument('--initializer', default='glorot')

    args, _ = parser.parse_known_args()
    return args

args = get_params()
params = vars(args)
SVD_PI = True
devices = ['0','1','-1']
if args.machine=='dgx1':
    devices = ['3','6','7']

real_device = args.device%len(devices)
os.environ["CUDA_VISIBLE_DEVICES"] = devices[real_device]
import tensorflow as tf

seed = args.seed
random.seed(args.seed)
np.random.seed(seed)
tf.random.set_seed(seed)

dtype = tf.float32
if args.dtype=='float64':
    dtype = tf.float64

eps = 1e-7
