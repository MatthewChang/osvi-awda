import numpy as np
import pickle as pkl
from pyutil import read_ims,write_sequence
import h5py
import os
import re
from matplotlib import pyplot as plt 
import pathlib
from einops import rearrange, reduce, repeat
from tqdm import tqdm
import cv2
import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('folder')
args = parser.parse_args()
    
from glob import glob
trajs = glob(f"{args.folder}/**/*.pkl",recursive=True)

tasks = os.listdir(args.folder)
for traj_path in tqdm(trajs):
    with open(traj_path, 'rb') as fil: tra = pkl.load(fil)
    obs = [obs for obs in tra['traj']]
    states = np.stack([x['obs']['ee_aa'] for x in obs])
    actions = np.stack([x['action'] for x in obs[1:]])
    diffs10 = states[10:,3]-states[:-10,3] #type: ignore
    # is commanded to close?
    p1 = actions[:,3] >= 0.1
    # is not closing enough over last d frames?
    p2 = diffs10 > -5e-2
    p2 = np.concatenate((p2,[p2[-1]]*9))
    # check not completely closed
    p3 = states[1:,3] > 0.2975
    grasp_frames = np.logical_and(p1,p2) #type: ignore
    grasp_frames = np.logical_and(grasp_frames,p3) #type: ignore
    smooth_factor = 6
    # smooth individual
    frames_valid = np.convolve(grasp_frames,[1]*smooth_factor,'valid') >= smooth_factor -2
    # frames_valid = np.convolve(grasp_frames,[1,1],'valid') >= 2
    grasps = np.concatenate((frames_valid,[frames_valid[-1]]*smooth_factor))
    if grasps.sum() > 0:
        start = np.where(grasps)[0].min()+4
        frames_new = np.zeros(states.shape[0]-1)
        frames_new[start:] = 1
        grasps = np.logical_and(frames_new,p1)
        grasps = np.concatenate((grasps,[grasps[-1]]))
    for i in range(len(grasps)):
        tra['traj']._data[i][0]['grasp'] = grasps[i]
    # write in-place
    pkl.dump(tra,open(traj_path,'wb'))
