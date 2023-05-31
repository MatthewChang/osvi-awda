import numpy as np
import gym
gym.logger.set_level(gym.logger.ERROR)
from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE)
from pyutil import LambdaWrapper, mujoco_depth_to_meters
from einops import rearrange, reduce, repeat
import gym.spaces as spaces

ALL_ENVS = sorted(['plate-slide-back-v2',
    'door-open-v2',
    'peg-unplug-side-v2',
    'stick-push-v2',
    'plate-slide-side-v2',
    'hammer-v2',
    'button-press-wall-v2',
    'coffee-button-v2',
    'basketball-v2',
    'drawer-close-v2',
    'soccer-v2',
    'plate-slide-v2',
    'handle-pull-v2',
    'window-open-v2',
    'handle-pull-side-v2',
    'coffee-push-v2',
    'window-close-v2',
    'hand-insert-v2',
    'coffee-pull-v2',
    'dial-turn-v2',
    'handle-press-side-v2',
    'push-back-v2',
    'bin-picking-v2',
    'disassemble-v2',
    'stick-pull-v2',
    'door-unlock-v2',
    'reach-wall-v2',
    'button-press-topdown-v2',
    'lever-pull-v2',
    'pick-place-v2',
    'pick-place-wall-v2',
    'peg-insert-side-v2',
    'button-press-topdown-wall-v2',
    'button-press-v2',
    'faucet-open-v2',
    'reach-v2',
    'handle-press-v2',
    'sweep-into-v2',
    'push-v2',
    'sweep-v2',
    'box-close-v2',
    'pick-out-of-hole-v2',
    'plate-slide-back-side-v2',
    'faucet-close-v2',
    'door-close-v2',
    'drawer-open-v2',
    'push-wall-v2',
    'assembly-v2',
    'door-lock-v2',
    'shelf-place-v2'])

num_indicators = 7
class MetaworldSingle(gym.Wrapper):
    def __init__(self, env,handcam=False,rgbd=False,resolution=224,depth_only=False,object_state=False,rgb=False):
        super().__init__(env)
        self.is_success=False
        self.handcam=handcam
        self.resolution=resolution
        self.rgbd = rgbd
        self.depth_only = depth_only
        self.render_depth = rgbd or depth_only
        self.object_state=object_state
        self.rgb = rgb
        self.waypoints = []
        if self.rgbd:
            raise Exception(f'not implemented')
        if self.rgb or self.depth_only or self.handcam:
            self.observation_space = spaces.Dict({
                "state": env.observation_space, 
                "img": spaces.Box(np.zeros((resolution,resolution,3)),np.ones((resolution,resolution,3))*255)
            })
        else:
            self.observation_space = spaces.Dict({
                "state": env.observation_space
            })

    def step(self,action):
        state,rew,done,info = self.env.step(action)
        self.is_success = self.is_success or info['success'] > 0.5
        info['is_success'] = self.is_success
        if self.object_state is False:
            state = state[:4]
        if self.rgbd or self.depth_only or self.rgb or self.handcam:
            state = {'img': self.get_visual_obs(),'state': state}
        else:
            state = {'state': state}
        return state,rew,done,info

    def set_waypoints(self,waypoints):
        self.waypoints = waypoints

    def process_depth(self,x):
        meters = mujoco_depth_to_meters(self.env.sim,x)
        return repeat(np.clip(meters,0,3)*255/3,'w h -> w h c',c = 3)

    def proc_img(self,x):
        if self.render_depth:
            meters_scaled = self.process_depth(x[1])
            if self.depth_only:
                x = meters_scaled
            else:
                x = np.concatenate((x[0],meters_scaled),axis=-1)
        return x

    def get_visual_obs(self):
        for wi in range(num_indicators):
            self.env.data.site_xpos[self.env.model.site_name2id(f'indicator{wi}')] = [0,0,0]
        if self.rgb or self.rgbd or self.depth_only:
            side = self.proc_img(self.env.sim.render(camera_name='corner_zoom', width=self.resolution, height=self.resolution, depth=self.render_depth))
        else:
            side = None
        if self.handcam:
            # modify camera position to be -0.06 in the metaworld directory
            # <camera name="gripperPOV" mode="track" pos="0 -0.06 0" quat="-1 -1.3 0 0" fovy="90" />
            hand = self.process_depth(self.env.sim.render(camera_name='gripperPOV', width=self.resolution, height=self.resolution, depth=True)[1])
            side = np.concatenate((side,hand),axis=-1) if side is not None else hand
        return side

    def render(self,mode='rgb'):
        for wi,waypoint in enumerate(self.waypoints):
            if waypoint[-1] > 0.1:
                self.env.data.site_xpos[self.env.model.site_name2id(f'indicator{wi}g')] = waypoint[:3]
            else:
                self.env.data.site_xpos[self.env.model.site_name2id(f'indicator{wi}')] = waypoint[:3]
        return self.env.sim.render(camera_name='corner_zoom', width=self.resolution, height=self.resolution)

    def reset(self, **kwargs):
        self.is_success = False
        res = self.env.reset(**kwargs)

        # from the metaworld env file, reset messes up site pos in renders, you need
        # to set the pos to fix it
        # see envs/mujoco/sawyer_xyz/sawyer_xyz_env.py
        # apparently you also need to render because the first render uses the old site positions
        # might be able to to just just after initilization instead of reset
        self.env.render('rgb_array')
        self.env.sim.forward()
        self.waypoints=[]
        for site in self.env.unwrapped._target_site_config: 
            self.env.unwrapped._set_pos_site(*site)
        if self.object_state is False:
            res = res[:4]
        if self.rgbd or self.depth_only or self.rgb or self.handcam:
            res = {'img': self.get_visual_obs(),'state': res}
        else:
            res = {'state': res}
        return res


class MetaworldGrasp(gym.Wrapper):
    def __init__(self, env,full_obs=False):
        super().__init__(env)
        self.init_o1 = None
        self.init_o2 = None
        self.full_obs = full_obs
        if self.full_obs:
            self.observation_space = env.observation_space
        else:
            if 'img' in env.observation_space.spaces.keys():
                self.slice = 'img'
                self.observation_space = env.observation_space['img']
            else:
                self.slice = 'state'
                self.observation_space = env.observation_space['state']
    def step(self,action):
        state,rew,done,info = self.env.step(action)
        pos = state['state'][:3]
        o1 = state['state'][4:7]
        o2 = state['state'][11:14]
        dist = min(np.linalg.norm(pos-o1),np.linalg.norm(pos-o2))
        rise = o1[2] - self.init_o1[2] + o2[2] - self.init_o2[2]
        success = False
        if dist > 0.05:
            # max of 1
            # print('reach')
            rew = -dist + state['state'][3]
        elif rise <= 0.05:
            # print('grasp')
            rew = 3 - state['state'][3] + max(rise*100,0)
        else:
            # print('lift')
            rew = 10
            success = True
        info['success'] = float(success)
        info['is_success'] = success
        if not self.full_obs:
            state = state[self.slice]
        return state,rew,done,info
    
    def reset(self, **kwargs):
        state = self.env.reset()
        start_range = spaces.Box(low=np.array([-0.1,-0.1,0.1]),high=np.array([0.1,0.1,0.2]))
        o1 = state['state'][4:7]
        # o2 = state['state'][11:14]
        start_pos = start_range.sample() + o1
        for _ in range(20): state,_,_,_ = self.env.step([0,0,1,-1])
        for _ in range(100):
            pos = state['state'][:3]
            diff = start_pos-pos
            if np.linalg.norm(diff) < 0.02: 
                break
            else:
                act = np.concatenate([np.clip(diff*10,-1,1),[-1]])
                state,_,_,_ = self.env.step(act)
        self.init_o1 = state['state'][4:7]
        self.init_o2 = state['state'][11:14]
        if not self.full_obs:
            state = state[self.slice]
        return state


class MetaworldGraspPoint(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.init_o1 = None
        self.init_o2 = None
        if 'img' in env.observation_space.spaces.keys():
            self.slice = 'img'
            self.observation_space = env.observation_space['img']
        else:
            self.slice = 'state'
            self.observation_space = env.observation_space['state']
        self.action_space = spaces.Box(low=np.array([-1,-1,-1]),high=np.array([1,1,1]))
    def step(self,efdelta):
        pos = self.env.unwrapped._get_obs()[:3]
        dest = efdelta+pos
        d1 = dest + [0,0,0.15]
        stable_frames = 0
        for _ in range(100):
            pos = self.env.unwrapped._get_obs()[:3]
            diff = d1 - pos
            action = np.concatenate((np.clip(diff*10,-1,1),[-1]))
            state,rew,done,info = self.env.step(action)
            if np.linalg.norm(diff) <= 0.06:
                stable_frames += 1
                if stable_frames > 10: 
                    break
            else:
                stable_frames = 0
        stable_frames = 0
        for _ in range(100):
            pos = self.env.unwrapped._get_obs()[:3]
            diff = dest - pos
            action = np.concatenate((np.clip(diff*10,-1,1),[-1]))
            state,rew,done,info = self.env.step(action)
            if np.linalg.norm(diff) <= 0.02:
                stable_frames += 1
                if stable_frames > 10: break
            else:
                stable_frames = 0
        for _ in range(30):
            state,rew,done,info = self.env.step([0,0,0,1])
        for _ in range(15):
            state,rew,done,info = self.env.step([0,0,1,1])
        state,rew,done,info = self.env.step([0,0,-1,1])
        pos = state['state'][:3]
        o1 = state['state'][4:7]
        o2 = state['state'][11:14]
        # dist = min(np.linalg.norm(pos-o1),np.linalg.norm(pos-o2))
        rise = o1[2] - self.init_o1[2] + o2[2] - self.init_o2[2]
        success = False
        if rise >= 0.05:
            rew = 1
            success = True
        else:
            rew = 0.5-np.linalg.norm(dest-o1)
        info['success'] = float(success)
        info['is_success'] = success
        return state[self.slice],rew,True,info
    
    def reset(self, **kwargs):
        state = self.env.reset()
        start_range = spaces.Box(low=np.array([-0.1,-0.1,0.1]),high=np.array([0.1,0.1,0.2]))
        o1 = state['state'][4:7]
        # o2 = state['state'][11:14]
        start_pos = start_range.sample() + o1
        for _ in range(20): state,_,_,_ = self.env.step([0,0,1,-1])
        for _ in range(100):
            pos = state['state'][:3]
            diff = start_pos-pos
            if np.linalg.norm(diff) < 0.02: 
                break
            else:
                act = np.concatenate([np.clip(diff*10,-1,1),[-1]])
                state,_,_,_ = self.env.step(act)
        self.init_o1 = state['state'][4:7]
        self.init_o2 = state['state'][11:14]
        return state[self.slice]

def make_metaworld_grasp(**kwargs):
    return MetaworldGraspPoint(make_metaworld_single(**kwargs))

# class MetaworldSingle(gym.Wrapper):
    # def __init__(self, env):
        # super().__init__(env)
        # self.is_success=False
    # def step(self,action):
        # state,rew,done,info = self.env.step(action)
        # self.is_success = self.is_success or info['success'] > 0.5
        # info['is_success'] = self.is_success
        # return state,rew,done,info
    # def reset(self, **kwargs):
        # self.is_success = False
        # return self.env.reset(**kwargs)
# def make_metaworld_single(task,test=False):
    # ml1 = metaworld.ML1(task)
    # env = ml1.train_classes[task]()
    # env = MetaworldSingle(env,ml1.test_tasks if test else ml1.train_tasks)
    # env = gym.wrappers.TimeLimit(env, max_episode_steps=env.max_path_length)
    # return env

def make_metaworld_single(env_name,seed=0,random_init=False,object_state=False,rgbd=False,depth_only=False,rgb=False,max_episode_steps=None,handcam=False):
    env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[f'{env_name}-goal-observable'](seed=seed)
    env._freeze_rand_vec = not random_init
    env = MetaworldSingle(env,rgbd=rgbd,depth_only=depth_only,object_state=object_state,rgb=rgb,handcam=handcam)
    max_episode_steps = max_episode_steps or env.max_path_length
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    return env

import time
if __name__ == '__main__':
    from matplotlib import pyplot as plt 
    # env = make_metaworld_single('reach-v2',depth_only=True)
    env = MetaworldGrasp(make_metaworld_single('sweep-into-v2',object_state=True,random_init=True,handcam=True),full_obs=True)
    state = env.reset()
    # plt.imsave('vis/test.png',state['img'][:,:,0]/255)
    # import pdb; pdb.set_trace()
    import imageio
    from policies.waypoints_policy import GraspPolicy
    for _ in range(10):
        ims = []
        state = env.reset()
        policy = GraspPolicy(panda=False,verbose=False)
        finished = False
        while not finished:
            state['img'] = state['img'][:,:,-1]
            ct = time.time()
            action,finished = policy.act(state)
            elapsed_time = time.time() - ct
            print(elapsed_time)
            state,rew,done,info = env.step(action)
            ims.append(env.render())
        print(info['success'])
        if info['success'] == 0:
            imageio.mimwrite('vis/test.mp4',ims,fps=30)

    # from matplotlib import pyplot as plt 
    # extent = env.sim.model.stat.extent
    # near = env.sim.model.vis.map.znear * extent
    # far = env.sim.model.vis.map.zfar * extent
    # meters = mujoco_depth_to_meters(env.sim,ims[:,:,3])
    # meters = np.clip(meters,0,3)
    # plt.clf()
    # plt.imshow(meters)
    # plt.colorbar()
    # plt.savefig('vis/test.png')
        
