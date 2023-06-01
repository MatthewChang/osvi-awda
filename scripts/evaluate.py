import numpy as np
import re
import torch
import argparse
from pyutil import *
import torch.utils.data
import argparse
import os
from dblog import DbLog
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from tqdm import tqdm
from scripts.eval_util import test_transformer
import time
import copy
from tensorboardX import SummaryWriter
# from metaworld.policies.policy import move

def get_remaining(dataset_number):
    _,nums = sorted_file_match(args.logdir,r'model_save-(\d+).pt')
    if dataset_number > 0:
        db_file = f'{args.logdir}/log{dataset_number}.db'
    else:
        db_file = f'{args.logdir}/log.db'
    dblog = DbLog(db_file)
    tag = 'eval/mean_reward_out_dist' if args.eval_only else 'eval/success_rate'
    sofar = np.array(dblog.read(tag)).astype(np.float32)
    if len(sofar) != 0:
        done = set(sofar[:,0].astype(int))
        nums = sorted(set(nums) - done)
    return nums

from hem.util import parse_basic_config
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('logdir')
    parser.add_argument('--bn',type=int)
    parser.add_argument('--motions',default=None,nargs='+')
    parser.add_argument('--write-images',action='store_true')
    parser.add_argument('--write-attention',action='store_true')
    parser.add_argument('--random-policy',action='store_true')
    parser.add_argument('--start',default=None,type=int)
    parser.add_argument('--rest',action='store_true')
    parser.add_argument('--watch',action='store_true')
    parser.add_argument('--eval-only',action='store_true')
    parser.add_argument('--instances',default=20,type=int)
    parser.add_argument('--envs',default=40,type=int)
    parser.add_argument('--dataset-number',default=0,type=int)
    parser.add_argument('--mod',default=None,type=int)
    parser.add_argument('--mod-eq',default=None,type=int)
    args = parser.parse_args()
    config_path = os.path.join(args.logdir, 'config.yaml')
    if not os.path.isfile(config_path):
        # it's the parent dir so select last experiment
        fils,nums = sorted_file_match(args.logdir,'bc.*?(\d+)')
        if len(fils) > 0:
            args.logdir = os.path.join(args.logdir,fils[-1])
            print(f'Loading last run {args.logdir}')
        config_path = os.path.join(args.logdir, 'config.yaml')
    if 'slurm' in args.logdir:
        os.system("sed -i 's@/projects/bbhm/mc48@${EXPERT_DATA}@' %s"%(config_path))
    os.system("sed -i 's@/dev/shm/mc48@${EXPERT_DATA}@' %s"%(config_path))
    config = parse_basic_config(config_path, resolve_env=True)
    files,nums = sorted_file_match(f"{args.logdir}",r'model_save-(\d+).pt')
    if args.rest:
        nums = get_remaining(args.dataset_number)
        if args.start is not None:
            nums = [ e for e in nums if e > args.start]
    elif args.start is not None:
        nums = [ e for e in nums if e > args.start]
    elif args.bn is not None:
        nums = [args.bn]
    if args.bn is None:
        writer = SummaryWriter(log_dir=os.path.join(args.logdir,'log'))
    else:
        writer=None
    print("WRITER: ",writer)
    if args.mod is not None:
        nums = [n for n in nums if n%args.mod == args.mod_eq ]
    print("evaluating: ",nums)
    if args.watch:
        while True:
            nums = get_remaining(args.dataset_number)
            print("found: ",nums)
            for n in nums:
                test_transformer(args.logdir,copy.deepcopy(config),n,writer,write_images = args.write_images,write_attention=args.write_attention,random_policy=args.random_policy,num_waypoints=None,instances=args.instances,eval_only=args.eval_only,motions=args.motions,num_envs=args.envs,dataset_number=args.dataset_number)
            print("waiting for more snapshots")
            time.sleep(60)
    else:
        for n in nums:
            test_transformer(args.logdir,copy.deepcopy(config),n,writer,write_images = args.write_images,write_attention=args.write_attention,random_policy=args.random_policy,num_waypoints=None,instances=args.instances,eval_only=args.eval_only,motions=args.motions,num_envs=args.envs,dataset_number=args.dataset_number)
