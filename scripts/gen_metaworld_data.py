from imageio.core.functions import mimwrite
import pathlib
import random
import os
from tqdm import tqdm
from matplotlib import pyplot as plt 
import itertools
import argparse
import numpy as np
from pyutil import *
import gym
from multiprocessing import Pool, set_start_method
import functools
from envs.metaworld_env import make_metaworld_single, ALL_ENVS
import h5py
import time
import pickle
from policies.waypoints_policy import WaypointsPolicy
import gym.spaces
from hem.datasets.savers.trajectory import Trajectory
import pickle as pkl
from scripts.eval_util import make_env
import metaworld.policies
from utils.projection_utils import embed_mat, image_point_to_pixels, do_projection
MVP_matrix = np.array([[-3.71769692e+02,  2.19356997e+02, 3.04825390e+01,  0.00000000e+00], [-1.56648501e+02, -1.56648501e+02, -3.71728228e+02,  0.00000000e+00], [-6.80413817e-01, -6.80413817e-01,  2.72165527e-01,  0.00000000e+00,]]) 
cam_pos = np.array([-1.1, -0.4,  0.6])

ost_projection = pkl.load(open('utils/transformation_mat_ost.pkl','rb'))
    # print('render ',ind,' ',traj_out_dir)
def rollout(args):
    env_name,ind,ost,trajectory_synthesis,out_dir = args
    # if ost:
        # out_dir = f'/data01/mc48/ost_high_ent2'
    # else:
        # out_dir = f'/data01/mc48/metaworld_handcam'
    if trajectory_synthesis:
        output = os.path.join(out_dir,env_name,'traj%03d.pkl'%(int(ind)))
    else:
        output = os.path.join(out_dir,env_name,'gripper','task_00','traj%03d.pkl'%(int(ind)))
    if os.path.isfile(output):
        print(f'skipping: {output}')
        return
    if ost:
        env = make_env('ost',sub_task=0)
        high=np.array([0.85602611, 0.6       , 1.07146709])
        low=np.array([ 0.35602611, -0.35      ,  0.84146709])
        assert trajectory_synthesis
    else:
        env = make_metaworld_single(env_name,random_init=True,rgb=True,object_state=True,handcam=True,seed=random.randint(0,int(1e6)))
        low = np.array([-0.34384765,  0.3501898 ,  0.04430117])
        high = np.array([0.25615235, 0.9001898 , 0.34430117])
    # obs = env.reset()
    # eef_pos = obs['state'][:3]
    # wp_space = gym.spaces.Box(low = np.array([-0.35,-0.25,-0.15]),high = np.array([0.25,0.3,0.15]))
    wp_space = gym.spaces.Box(low = low,high = high)
    waypoints = [wp_space.sample() for _ in range(random.randint(1,3))]
    waypoints = np.stack([np.concatenate((wp,[0])) for wp in waypoints])
    def get_policy(env_name):
        name = ''.join([ e.capitalize() for e in env_name.split('-')])
        if env_name == 'peg-insert-side-v2': 
          return metaworld.policies.SawyerPegInsertionSideV2Policy()
        else:
          return getattr(metaworld.policies,f'Sawyer{name}Policy')()
    if trajectory_synthesis:
        policy = WaypointsPolicy(waypoints,panda=ost,base_actions=ost)
        act = policy.act
    else:
        policy = get_policy(env_name)
        def act(state):
          action = policy.get_action(state['state'])
          return action,False
    ct = time.time()
    while True:
        states,ims,actions,rewards,infos = rollout_general(env,act,render_ims=False)
        ims = np.stack([s['img'][...,:3] for s in states])
        handcam = np.stack([s['img'][...,3] for s in states])
        states = np.stack([s['state'] for s in states])
        actions = np.array(actions)
        if not trajectory_synthesis:
            # only save trajectories which achieve success with some frames padded on the end
            success = [ e['success'] for e in infos]
            sucinds = np.where(np.array(success) == 1)[0]
            if len(sucinds) == 0 and not trajectory_synthesis:
                print('t0: ',time.time() - ct)
                continue
            stop_ind = sucinds.min()+1+5 
            ims,handcam,states,actions = [ e[:stop_ind] for e in (ims,handcam,states,actions)]
        actions = actions[:-1]
        if len(ims) < 6:
            continue
        print('t1: ',time.time() - ct)
        traj = Trajectory()
        if ost:
            proj = lambda point: image_point_to_pixels(ost_projection@embed_mat(point),ims[0].shape)
        else:
            proj = lambda point: do_projection(point,MVP_matrix,cam_pos,frame_height=ims[0].shape[0],frame_width=ims[0].shape[1])
        point = proj(states[0,:3])
        grasp_val = 0 if trajectory_synthesis else None
        traj.append({'image': ims[0],'full_state':states[0],'ee_aa':states[0,:4],'grasp':grasp_val,'eef_point':point,'hand_cam':handcam[0]})

        for i in range(actions.shape[0]):
            point = proj(states[i+1,:3])
            traj.append({'image': ims[i+1],'grasp':grasp_val,'full_state':states[i+1],'ee_aa':states[i+1,:4],'eef_point':point,'hand_cam': handcam[i+1],},action=actions[i],raw_state=states[i+1])
                
        pathlib.Path(os.path.dirname(output)).mkdir(parents=True, exist_ok=True)
        print(f"writing {output}")
        pkl.dump({'traj':traj,'env_type': 'metaworld'},open(output,'wb'))
        print('t2: ',time.time() - ct)
        env.close()
        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--debug',action='store_true')
    parser.add_argument('--ost',action='store_true')
    parser.add_argument('--trajectory-synthesis',action='store_true')
    parser.add_argument('--output')
    parser.add_argument('--end',type=int)
    parser.add_argument('--start',type=int)
    args = parser.parse_args()
    set_start_method('spawn', force=True)
    if args.ost:
        envs = ["ost"]
    else:
        envs = ALL_ENVS
    fargs = list(itertools.product(envs,range(args.start,args.end)))
    fargs = [tup+(args.ost,args.trajectory_synthesis,args.output) for tup in fargs]
    if args.debug:
        for farg in fargs:
            rollout(farg)
    pool = Pool(processes=25)
    for _ in tqdm(pool.imap_unordered(rollout,fargs),total=len(fargs)):
        pass
    pool.close()
    pool.join()
    os._exit(0)
