from pyutil import write_sequence
from robosuite_mosaic import load_controller_config
from robosuite_env.controllers.expert_basketball import \
    get_expert_trajectory as basketball_expert
from robosuite_env.controllers.expert_nut_assembly import \
    get_expert_trajectory as nut_expert
from robosuite_env.controllers.expert_pick_place import \
    get_expert_trajectory as place_expert 
from robosuite_env.controllers.expert_block_stacking import \
    get_expert_trajectory as stack_expert
from robosuite_env.controllers.expert_drawer import \
    get_expert_trajectory as draw_expert
from robosuite_env.controllers.expert_button import \
    get_expert_trajectory as press_expert
from robosuite_env.controllers.expert_door import \
    get_expert_trajectory as door_expert
import numpy as np
import cv2
import gym
from matplotlib import pyplot as plt 
import functools
import os
import pickle as pkl
import json
import random
import torch
from os.path import join 
from multiprocessing import Pool, cpu_count
import imageio
from utils.projection_utils import image_point_to_pixels, pixels_to_image_point, embed_mat,embed_matrix_square
from envs.metaworld_env import make_metaworld_single, ALL_ENVS
import robosuite_mosaic.utils.transform_utils as T
from pyutil import mujoco_depth_to_meters
from einops import rearrange, reduce, repeat

def process_depth(sim, x):
    meters = mujoco_depth_to_meters(sim, x)
    return repeat(np.clip(meters, 0, 3) * 255 / 3, "w h -> w h c", c=3)

TASK_ENV_MAP = {
    'door':     {
        'n_task': 4,
        'env_fn': door_expert,
        'panda':  'PandaDoor',
        'sawyer': 'SawyerDoor',
        },
    'drawer': {
        'n_task':   8, 
        'env_fn':   draw_expert,
        'panda':    'PandaDrawer',
        'sawyer':   'SawyerDrawer',
        },
    'basketball': {
        'n_task':   12, 
        'env_fn':   basketball_expert,
        'panda':    'PandaBasketball',
        'sawyer':   'SawyerBasketball',
        },
    'nut_assembly':  {
        'n_task':   9, 
        'env_fn':   nut_expert,
        'panda':    'PandaNutAssemblyDistractor',
        'sawyer':   'SawyerNutAssemblyDistractor',
        },
    'stack_block': {
        'n_task':   6, 
        'env_fn':   stack_expert,
        'panda':    'PandaBlockStacking',
        'sawyer':   'SawyerBlockStacking',
        },
    'pick_place': {
        'n_task':   16, 
        'env_fn':   place_expert,
        'panda':    'PandaPickPlaceDistractor',
        'sawyer':   'SawyerPickPlaceDistractor',
        },
    'button': {
        'n_task':   6, 
        'env_fn':   press_expert,
        'panda':    'PandaButton',
        'sawyer':   'SawyerButton',
        },
    'stack_new_color': {
        'n_task':   6, 
        'env_fn':   stack_expert,
        'panda':    'PandaBlockStacking',
        'sawyer':   'SawyerBlockStacking',
        },
    'stack_new_shape': {
        'n_task':   6, 
        'env_fn':   stack_expert,
        'panda':    'PandaBlockStacking',
        'sawyer':   'SawyerBlockStacking',
        },
}

def make_env(name,sawyer=False,task=None,seed=0):
    arm = 'saywer' if sawyer else 'panda'
    specs = TASK_ENV_MAP[name]
    env_name = specs.get(arm)
    env_fn = specs['env_fn']
    config = load_controller_config(default_controller='IK_POSE_IMPEDENCE')
    if task is None:
        task = random.randint(0,specs['n_task']-1)
    env = env_fn(env_name, controller_type=config, camera_obs=True, task=task, seed=seed, env_seed=0,heights=100,widths=180,ret_env=True)
    return env

import gym.spaces as spaces
def make_env_eval(name,sawyer=False,task=None,handcam=False,seed=0):
    env = MosaicEval(make_env(name,sawyer,task,seed=seed),handcam=handcam)
    # env.observation_space = spaces.Dict({
        # "state": spaces.Box(np.zeros((8,)),np.ones((8,))*10), 
        # "img": spaces.Box(np.zeros((100,180,3)),np.ones((100,180,3))*255)
    # })
    # env.action_space = spaces.Box(np.ones((7,))*-10,np.ones((7,))*10) 
    return env

from robosuite_env.custom_ik_wrapper import normalize_action
class MosaicEval(gym.Wrapper):
    def __init__(self, env,handcam=False):
        env.observation_space = spaces.Dict({
            "state": spaces.Box(np.zeros((8,)),np.ones((8,))*10), 
            "img": spaces.Box(np.zeros((100,180,3)),np.ones((100,180,3))*255)
        })
        env.action_space = spaces.Box(np.ones((7,))*-10,np.ones((7,))*10) 
        env.reward_range = [0,1]
        env.metadata = ""
        super().__init__(env)
        self.is_success=False
        self.handcam=handcam
        self.base_quat = self.env._eef_xquat

    def get_handcam(self):
        return process_depth(self.env.sim,
            self.env.sim.render(
                camera_name="robot0_gripperPOV",
                width=180,
                height=100,
                depth=True,
            )[1]
        )

    def step(self,action,rot=None):
        cur_pos = self.env._get_observation()["eef_pos"]
        if action.shape[-1] not in [4,7]:
            raise Exception(f'bad shape')
        if action.shape[-1] == 7:
            rot = action[3:6]
        if rot is None:
            rot = T.quat2axisangle(self.base_quat)

        # first 3 are desired absolute eef pos in world coords
        # 3:7 should be desired rotation relative to world rotation frame as angle axis
        # angle first then 3 axis values
        action = np.concatenate(
                [action[:3] + cur_pos, rot, action[-1:]]
        )
        norm_action = normalize_action(action,self.env.ranges)
        state,rew,done,info = self.env.step(norm_action)
        self.is_success = self.is_success or rew >= 1
        info['is_success']=self.is_success
        if self.handcam:
            state["image"] = np.concatenate((state['image'],self.get_handcam()),axis=-1)
        return self.proc_obs(state),rew,done,info

    def proc_obs(self,obs):
        return {'img': obs['image'],'state': np.concatenate((obs['eef_pos'],obs['eef_quat'],obs['gripper_qpos'][0:1]))}
    def render(self,_='rgb'):
        return self.env.render()

    def reset(self, **kwargs):
        res = self.env.reset(**kwargs)
        self.is_success = False
        self.base_quat = self.env._eef_xquat
        if self.handcam:
            res["image"] = np.concatenate((res['image'],self.get_handcam()),axis=-1)
        obs = self.proc_obs(res)
        return obs

"""
gpu 0 ipython -i /home/mc48/one_shot_transformers/envs/mosaic_env.py
"""

door_max = np.array([0.10200438, 0.50929266, 1.3965743 ], dtype=np.float32)
door_min = np.array([-0.18456243, -0.49024346,  0.9742076 ], dtype=np.float32)
basketball_max = np.array([0.23629479, 0.42231768, 1.384172  ], dtype=np.float32)
basketball_min = np.array([-0.2389883 , -0.3853428 ,  0.82498914], dtype=np.float32)

default_rot = np.array([np.pi,0,0])
# base = T.axisangle2quat(default_rot)
# mod = T.axisangle2quat([0,0,np.pi*0.5])
# rot = T.quat2axisangle(T.quat_multiply(base,mod))
# button_rot computation
button_rot = np.array([ 2.2214415e+00, -2.2214415e+00,  1.3602407e-16])

from policies.waypoints_policy import action_toward
# test IK performance
if __name__ == '__main__':
    env = make_env_eval('button',handcam=True)
    obs = env.reset()
    spos = obs['state'][:3]
    ims = []
    for _ in range(10):
        action = action_toward(obs['state'][:3],spos,scale=1)
        obs,_,_,_ = env.step(np.concatenate((action[:3],default_rot,action[-1:])),rot=button_rot)
        ims.append(env.render())
    imageio.mimwrite('vis/test.mp4',ims)
    import pdb; pdb.set_trace()

    env = make_env_eval('door',handcam=True)
    # env = make_env_eval('basketball')
    obs = env.reset()
    spos = obs['state'][:3]
    door_pos = env.sim.data.site_xpos[env.sim.model.site_name2id('door1_handle')]
    door_pos += [0.05,0,0]
    # env.sim.model.site_names
    from matplotlib import pyplot as plt 
    ims=[]
    for _ in range(3):
        action = action_toward(obs['state'][:3],spos,scale=1)
        obs,_,_,_ = env.step([1,0,0,0])
        ims.append(env.render())
    for _ in range(5):
        action = action_toward(obs['state'][:3],spos,scale=1)
        obs,_,_,_ = env.step([0,1,0,0])
        ims.append(env.render())
    for _ in range(50):
        action = action_toward(obs['state'][:3],spos,scale=1)
        obs,_,_,_ = env.step(action)
        ims.append(env.render())
    ims = np.stack(ims)
    imageio.mimwrite('vis/test.mp4',ims)

    # import pdb; pdb.set_trace()

    # import mujoco_py
    # viewer = mujoco_py.MjRenderContextOffscreen(env.env.sim, -1)
    # viewer.add_marker(pos=spos,label='test')
    # frame = viewer.read_pixels(224,224,False,False)
    # frame
    # env.set_waypoints([spos,spos+T.quat2mat(env.env._eef_xquat)@[0.01,0,0]])
    # env.set_waypoints([spos,spos+[0.2,0.00,0.1]])

    # #rotation thinks +z is towards, y is up, x is left right
    # plt.imsave('vis/test.png',env.render())

    # # position
    # # +x is towards, +y is right, +z is up
    # # maybe axes are messed up some how? rotate round x doesn't seem to be doing the right thing
    # T.quat2mat(T.quat_multiply(T.quat_inverse(base_quat),env.env._eef_xquat))

    # T.quat2axisangle(T.quat_inverse(base_quat),env.env._eef_xquat)
    # for x in np.linspace(0,1,num=10):
        # print(T.quat2axisangle(T.quat_slerp(env.env._eef_xquat,base_quat,x,shortestpath=False)))
    # start_aa

    # from robosuite_env.custom_ik_wrapper import normalize_action
    # norm_action = normalize_action(dest,env.ranges)
    # obs2,_,_,_ = env.step(norm_action)
    # im2 = env.sim.render(camera_name='agentview',width=180,height=100)
    # plt.imsave('vis/test.png',obs2['image'])
    # plt.imsave('vis/test2.png',obs['image'])


