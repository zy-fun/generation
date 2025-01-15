import argparse
import os
import torch
from exp.exp import Exp
import random
import numpy as np

if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Trajectory Generation')
    parser.add_argument('--task_name', type=str, required=False, default='default',
                        help='task name, options:[]')
    parser.add_argument('--batch_size', type=int, required=False, default=1,
                        help='batch size')
    parser.add_argument('--data', type=str, required=False, default='shenzhen_20201104',
                        help='data name')
    
    parser.add_argument('--learning_rate', type=float, required=False, default=0.001,)
    parser.add_argument('--train_epochs', type=int, required=False, default=10,)
    args = parser.parse_args()
    # args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    # args.use_gpu = True if torch.cuda.is_available() else False

    print(torch.cuda.is_available())

    print('Args in experiment:')
    # print_args(args)

    # setting record of experiments
    exp = Exp(args)  # set experiments

    print('>>>>>>>start training : >>>>>>>>>>>>>>>>>>>>>>>>>>')
    exp.train()

    print('>>>>>>>testing : <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.test()
    torch.cuda.empty_cache()