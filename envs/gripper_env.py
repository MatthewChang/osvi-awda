import copy
import numpy as np
from hem.robosuite.controllers.expert_pick_place import get_expert_trajectory
import gym.spaces as spaces
import gym.envs
from gym import ObservationWrapper
from gym.wrappers import FrameStack
import torch
import gym
from gym import spaces
import torchvision.transforms as transforms
from pyutil import mujoco_depth_to_meters
from einops import rearrange, reduce, repeat

torch.set_num_threads(1)
from hem.robosuite.custom_ik_wrapper import normalize_action

class GripperEnv(gym.Env):
    metadata = {"render.modes": ["rgb"]}

    def __init__(
        self,
        task=None,
        eval=False,
        rgb=False,
        depth=False,
        object_state=False,
        seed=0,
        handcam=False,
    ):
        super(GripperEnv, self).__init__()
        if task is None:
            raise Exception(f"No task specified")
        self.eval = eval
        self.env = get_expert_trajectory(
            "PandaPickPlaceDistractor", task=int(task), ret_env=True, seed=seed
        )
        self.observation_space = spaces.Box(-10, 10, shape=(23,))
        self.action_space = spaces.Box(-10, 10, (8,))
        self.success = False
        self.max_T = 99
        self.n_steps = 0
        self.rgb = rgb
        self.depth = depth
        # self.resolution = 224
        self.handcam = handcam
        self.object_state = object_state
        self.waypoints = []
        if self.rgb or self.depth or self.handcam:
            self.observation_space = spaces.Dict(
                {
                    "state": self.observation_space,
                    "img": spaces.Box(
                        np.zeros((240, 320, 3)),
                        np.ones((240, 320, 3)) * 255,
                    ),
                }
            )
        else:
            self.observation_space = spaces.Dict({"state": self.observation_space})

    def process_depth(self, x):
        meters = mujoco_depth_to_meters(self.env.sim, x)
        return repeat(np.clip(meters, 0, 3) * 255 / 3, "w h -> w h c", c=3)

    def step(self, action,debug=False):
        if len(action) == 4:
            # fill in rotation if given ee and grip
            cur_pos = self.env.unwrapped._get_observation()["eef_pos"]
            action = np.concatenate(
                [action[:3] + cur_pos, [0.296875, 0.703125, 0.703125, 0.0], action[-1:]]
            )
            action = normalize_action(action)
            # mimic hardcoding from ost paper, fixes orientation of gripper
        action[3:7] = [0.296875, 0.703125, 0.703125, 0.0]
        # hardcode grip clamping like they do
        action[-1] = 1 if action[-1] > 0 and self.n_steps < self.max_T else -1
        obs, rew, done, info = self.env.step(action)
        self.success |= rew > 0
        done = done or rew > 0
        info["success"] = self.success
        if info['success']:
            info['success_frame'] = obs
        info["is_success"] = info["success"]
        self.n_steps += 1
        return self.obs_to_output(obs), rew, done, info

    def obs_to_output(self,obs):
        # keys = ["eef_pos", "Milk0_pos", "Bread0_pos", "Cereal0_pos", "Can0_pos"]
        # state = np.concatenate([obs[x] for x in keys])
        keys = ['Milk0_pos', 'Bread0_pos', 'Cereal0_pos', 'Can0_pos']
        eefstate = np.concatenate((obs['ee_aa'][:3],obs['gripper_qpos'][:1]))
        state = np.concatenate([eefstate]+[obs[x] for x in keys])
        if self.object_state:
            out_state = {"state": state}
        else:
            out_state = {"state": state[:4]}
        if self.rgb:
            out_state["img"] = obs['image']
        if self.handcam:
            out_state["img"] = np.concatenate((out_state['img'],self.get_handcam()),axis=-1) if 'img' in out_state else self.get_handcam()
        return out_state

    def set_waypoints(self, waypoints):
        self.waypoints = waypoints
    def get_handcam(self):
        return self.process_depth(
            self.env.sim.render(
                camera_name="gripperPOV",
                width=320,
                height=240,
                depth=True,
            )[1]
        )

    def reset(self):
        obs = self.env.reset()
        self.success = False
        self.n_steps = 0
        return self.obs_to_output(obs)

    def render(self, mode="rgb", close=False):
        for i, wp in enumerate(self.waypoints):
            if wp[-1] > 0.1:
                self.unwrapped.env.sim.data.site_xpos[self.unwrapped.env.sim.model.site_name2id(f'indicator{i}g')] = wp[:3]
            else:
                self.unwrapped.env.sim.data.site_xpos[self.unwrapped.env.sim.model.site_name2id(f'indicator{i}')] = wp[:3]
        camera_obs = self.env.sim.render(
            camera_name=self.env.camera_name,
            width=self.env.camera_width,
            height=self.env.camera_height,
        )
        camera_obs = camera_obs[80:,::-1]
        return camera_obs


from gym.spaces import Box


class LambdaWrapper(ObservationWrapper):
    r"""Observation wrapper that runs a function. The observation_space property won't reflect scale changes."""

    def __init__(self, env, lam):
        super().__init__(env)
        self.lam = lam
        self.observation_space = Box(
            lam(env.observation_space.low), lam(env.observation_space.high)
        )

    def observation(self, state):
        return self.lam(state)


def FlatFrameStack(env, num):
    return LambdaWrapper(FrameStack(env, num_stack=num), np.ravel)


# def make_gripper_env(**kwargs):
# env = GripperEnv(task=task,eval=eval)
# return env



def register_envs():
    gym.envs.register(
        id="Gripper-v0",
        entry_point="envs.gripper_env:GripperEnv",
        max_episode_steps=200,
    )
from policies.waypoints_policy import GraspPolicy
if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import imageio

    # env.reset()
    # state,_,_,_ = env.step([0,0,0,0])
    # env.unwrapped.env.sim.forward()
    # env.unwrapped.env.unwrapped.sim.data.set_joint_qpos( f"indicator1g", np.concatenate((state['state'][:3], [0] * 4)))
    # env.unwrapped.env.unwrapped.sim.data.set_joint_qpos( f"indicator1g", np.concatenate(([0,0,0], [0] * 4)))
    # env.unwrapped.env.sim.data.site_xpos[env.unwrapped.env.sim.model.site_name2id(f'indicator0')] = state['state'][:3]
    # env.unwrapped.env.sim.forward()
    # state,_,_,_ = env.step([0,0,0,0])
    # state,_,_,_ = env.step([0,0,0,0],debug=True)
    # plt.imsave('vis/test.png',env.render())
    # import pdb; pdb.set_trace()
    # exit()
    # env.unwrapped.env.unwrapped.sim.forward()
    # env.unwrapped.env.unwrapped.sim.step()
    env = GripperEnv(task=0, object_state=True,rgb=True,handcam=True)
    state = env.reset()
    plt.imsave('vis/test.png',env.render())
    # state,_,_,_ = env.step([0,0,0,0])
    for _ in range(10):
        # obj = state["state"][4:7] + [0, 0.05, 0.15]
        obj = state["state"][13:16] + [-0.6, -0.6, 0.15]
        pos = state["state"][:3]
        state, _, _, _ = env.step([*(obj - pos), -1])
        # obj - pos
        # np.linalg.norm(obj - pos)
    plt.imsave('vis/test.png',env.render())
    # state, _, _, _ = env.step([0.1,0,0, -1])
    # obj = state["state"][3:6] + [0, 0, 0.15]
    # pos = state["state"][:3]
    plt.imsave('vis/test.png',env.render())
    policy = GraspPolicy(panda=True,verbose=True)
    imgs = []
    done = False
    while not done:
        # action,done = policy.act(state)
        st = state['state']
        im = state['img']
        action,done = policy.act({'state':st,'img':im[:,:,-1]})
        state, _, _, _ = env.step(action)
        rend = np.concatenate((env.render(),im[...,-3:]),axis=1)
        imgs.append(rend)
    imageio.mimwrite('vis/test.mp4',imgs,fps=10)
    print(len(imgs))
