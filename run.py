import argparse
import os
import torch
from exp.exp import Exp
from exp.exp_avg import Exp_AVG
import random
import numpy as np
import time

if __name__ == '__main__':
    time_now = time.time()
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Trajectory Generation')
    # basic config
    parser.add_argument('--task_name', type=str, required=False, default='exp',
                        help='task name')
    parser.add_argument('--model', type=str, required=False, default='Transformer',
                        help='model name')
    parser.add_argument('--test', required=False, default=False, action='store_true',)
    parser.add_argument('--load_model_path', type=str, required=False, default='checkpoints/2025-01-22_15-47-34/checkpoint.pth', help='only available when test is True')
    parser.add_argument('--avg_by_hour', required=False, default=False, action='store_true', help='only available when task_name == exp_avg')
    
    # data loader
    parser.add_argument('--data', type=str, required=False, default='shenzhen_20201104',
                        help='data name')
    parser.add_argument('--use_subset', required=False, default=False, action='store_true', help='only available when task_name == exp')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    
    # model define
    parser.add_argument('--d_model', type=int, required=False, default=512)
    parser.add_argument('--d_ff', type=int, required=False, default=2048)
    parser.add_argument('--n_heads', type=int, required=False, default=8)
    parser.add_argument('--e_layers', type=int, required=False, default=2)
    parser.add_argument('--d_layers', type=int, required=False, default=1)
    parser.add_argument('--dropout', type=float, required=False, default=0.1)
    parser.add_argument('--activation', type=str, required=False, default='gelu')
    parser.add_argument('--use_pretrained', type=bool, required=False, default=True, help='use pretrained embeddings or not')

    parser.add_argument('--enc_emb', type=int, required=False, default=128, help='encoder embedding size, only functional when use_pretrained is False')
    parser.add_argument('--enc_dim', type=int, required=False, default=2)
    parser.add_argument('--dec_dim', type=int, required=False, default=4)
    parser.add_argument('--c_out', type=int, required=False, default=1)
    parser.add_argument('--n_vocab', type=int, required=False, default=27411, help='vocab size, only functional when use_pretrained is False')

    # optimization
    parser.add_argument('--batch_size', type=int, required=False, default=16,
                        help='batch size')
    parser.add_argument('--learning_rate', type=float, required=False, default=0.001,)
    parser.add_argument('--train_epochs', type=int, required=False, default=10,)
    

    args = parser.parse_args()
    # args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    # args.use_gpu = True if torch.cuda.is_available() else False

    print(torch.cuda.is_available())

    print('Args in experiment:')
    print(args)

    # setting record of experiments
    exp_dict = {
        'exp': Exp,
        'exp_avg': Exp_AVG,
    }
    exp = exp_dict[args.task_name](args)  # set experiments

    if not args.test:
        print('>>>>>>>start training : >>>>>>>>>>>>>>>>>>>>>>>>>>')
        exp.train()
    else:
        exp.model.load_state_dict(torch.load(args.load_model_path))
    
    print('>>>>>>>testing : <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.test()
    torch.cuda.empty_cache()
    total_time = time.time() - time_now
    print('total time: {:.4f}s'.format(total_time))