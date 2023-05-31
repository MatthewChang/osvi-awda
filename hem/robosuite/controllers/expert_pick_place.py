from pyquaternion import Quaternion
from robosuite.environments.baxter import BaxterEnv
from robosuite.environments.sawyer import SawyerEnv
from robosuite.environments.panda import PandaEnv
import robosuite
import os
import numpy as np
from hem.robosuite import get_env
from hem.datasets import Trajectory
import pybullet as p
from pyquaternion import Quaternion
import random
from hem.robosuite.custom_ik_wrapper import normalize_action


def _clip_delta(delta, max_step=0.015):
    norm_delta = np.linalg.norm(delta)

    if norm_delta < max_step:
        return delta
    return delta / norm_delta * max_step


class PickPlaceController:
    def __init__(self, env, tries=0,bad=False):
        assert env.single_object_mode == 2, "only supports single object environments at this point!"
        self._env = env
        self._g_tol = 5e-2 ** (tries + 1)
        self.bad = bad
        self.reset()
        
    def _calculate_quat(self, angle):
        if isinstance(self._env, SawyerEnv):
            new_rot = np.array([[np.cos(angle), -np.sin(angle), 0],[np.sin(angle), np.cos(angle), 0],[0, 0, 1]])
            return Quaternion(matrix=self._base_rot.dot(new_rot))
        return self._base_quat
    
    def reset(self):
        object_id = self._env.object_id
        if self.bad:
            object_id = (object_id + 1) % 4
        self._object_name = self._env.item_names_org[object_id] + '0'
        self._target_loc = self._env.target_bin_placements[object_id] + [0, 0, 0.3]
        # TODO this line violates abstraction barriers but so does the reference implementation in robosuite
        self._jpos_getter = lambda : np.array(self._env._joint_positions)
        self._clearance = 0.03 if 'Milk' not in self._object_name else -0.01

        if isinstance(self._env, SawyerEnv):
            self._obs_name = 'eef_pos'
            self._default_speed = 0.15
            self._final_thresh = 1e-2
            self._base_rot = np.array([[0, 1, 0.], [1, 0, 0.], [0., 0., -1.]])
            self._base_quat = Quaternion(matrix=self._base_rot)
        elif isinstance(self._env, PandaEnv):
            self._obs_name = 'eef_pos'
            self._default_speed = 0.15
            self._final_thresh = 6e-2
            self._base_rot = np.array([[0, 1, 0.], [1, 0, 0.], [0., 0., -1.]])
            self._base_quat = Quaternion(matrix=self._base_rot)
        else:
            raise NotImplementedError

        self._t = 0
        self._intermediate_reached = False
        self._hover_delta = 0.2
        

    def _get_target_pose(self, delta_pos, base_pos, quat, max_step=None):
        if max_step is None:
            max_step = self._default_speed

        delta_pos = _clip_delta(delta_pos, max_step)
        quat = np.concatenate(([quat.angle / np.pi], quat.axis))
        if quat[0] < 0:
            quat[0] += 1
        return normalize_action(np.concatenate((delta_pos + base_pos, quat)))

    def act(self, obs):
        if self._t == 0:
            self._start = -1
            try:
                y = -(obs['{}_pos'.format(self._object_name)][1] - obs[self._obs_name][1])
                x = obs['{}_pos'.format(self._object_name)][0] - obs[self._obs_name][0]
            except:
                import pdb; pdb.set_trace()
            angle = np.arctan2(y, x) - np.pi/3 if 'Cereal' in self._object_name else np.arctan2(y, x)
            self._target_quat = self._calculate_quat(angle)

        if self._start < 0 and self._t < 15:
            if np.linalg.norm(obs['{}_pos'.format(self._object_name)] - obs[self._obs_name] + [0, 0, self._hover_delta]) < self._g_tol or self._t == 14:
                self._start = self._t
                self._rise_t = self._start + 8

            quat_t = Quaternion.slerp(self._base_quat, self._target_quat, min(1, float(self._t) / 5))
            eef_pose = self._get_target_pose(obs['{}_pos'.format(self._object_name)] - obs[self._obs_name] + [0, 0, self._hover_delta], obs['eef_pos'], quat_t)
            action = np.concatenate((eef_pose, [-1]))
        elif self._t < self._start + 10: 
            if self._t  < self._rise_t:
                eef_pose = self._get_target_pose(obs['{}_pos'.format(self._object_name)] - obs[self._obs_name] - [0, 0, self._clearance], obs['eef_pos'], self._target_quat)  
                action = np.concatenate((eef_pose, [-1]))
            else:
                eef_pose = self._get_target_pose(obs['{}_pos'.format(self._object_name)] - obs[self._obs_name] + [0, 0, self._hover_delta], obs['eef_pos'], self._target_quat)
                action = np.concatenate((eef_pose, [1]))

        elif np.linalg.norm(self._target_loc - obs[self._obs_name]) > self._final_thresh: 
            target = self._target_loc        
            eef_pose = self._get_target_pose(target - obs[self._obs_name], obs['eef_pos'], self._target_quat)
            action = np.concatenate((eef_pose, [1]))
        else:
            eef_pose = self._get_target_pose(np.zeros(3), obs['eef_pos'], self._target_quat)
            action = np.concatenate((eef_pose, [-1]))
        
        self._t += 1
        return action

    def disconnect(self):
        p.disconnect()


def get_expert_trajectory(env_type, camera_obs=True, renderer=False, task=None, ret_env=False, seed=None, force_success=False, env_seed=None, depth=False,hand_depth=False):
    # raise Exception(f'not using this version')
    # print('HERE')
    seed = seed if seed is not None else random.getrandbits(32)
    env_seed = seed if env_seed is None else env_seed
    seed_offset = sum([int(a) for a in bytes(env_type, 'ascii')])
    np.random.seed(env_seed)

    success, use_object = False, ''
    if task is not None:
        assert 0 <= task <= 15, "task should be in [0, 15]"
        use_object = int(task // 4)
        rg, db, = False, task % 4
    else:
        rg, db = True, None

    if ret_env:
        env = get_env(env_type, force_object=use_object, randomize_goal=rg, default_bin=db, has_renderer=renderer, reward_shaping=False, use_camera_obs=camera_obs, camera_height=320, camera_width=320, camera_depth=depth,hand_depth=hand_depth)
        return env

    tries = 0
    while not success:
        np.random.seed(env_seed)
        env = get_env(env_type, force_object=use_object, randomize_goal=rg, default_bin=db, has_renderer=renderer, reward_shaping=False, use_camera_obs=camera_obs, camera_height=320, camera_width=320, force_success=force_success, camera_depth=depth,hand_depth=hand_depth)
        controller = PickPlaceController(env.env, tries=tries)
        np.random.seed(seed + int(tries // 3) + seed_offset)
        obs = env.reset()
        mj_state = env.sim.get_state().flatten()
        sim_xml = env.model.get_xml()
        traj = Trajectory(sim_xml)

        env.reset_from_xml_string(sim_xml)
        env.sim.reset()
        env.sim.set_state_from_flattened(mj_state)
        env.sim.forward()
        use_object = env.object_id

        traj.append(obs, raw_state=mj_state)
        for _ in range(int(env.horizon // 10)):
            action = controller.act(obs)
            obs, reward, done, info = env.step(action)
            if renderer:
                env.render()

            mj_state = env.sim.get_state().flatten()
            traj.append(obs, reward, done, info, action, mj_state)
            
            if reward:
                success = True
                break
        tries += 1

    if renderer:
        env.close()
    
    controller.disconnect()
    del controller
    del env
    return traj
