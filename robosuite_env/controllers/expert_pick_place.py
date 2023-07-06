import sys
from pathlib import Path

if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))
import numpy as np
from robosuite_env import get_env
from mosaic.datasets import Trajectory
import pybullet as p
from pyquaternion import Quaternion
import random
from robosuite_env.custom_ik_wrapper import normalize_action
from robosuite_mosaic import load_controller_config
from robosuite_mosaic.utils.transform_utils import quat2axisangle
from robosuite_mosaic.utils import RandomizationError
import torch
import os
import mujoco_py

# in case rebuild is needed to use GPU render: sudo mkdir -p /usr/lib/nvidia-000
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000
# pip uninstall mujoco_py; pip install mujoco_py 

def _clip_delta(delta, max_step=0.015):
    norm_delta = np.linalg.norm(delta)

    if norm_delta < max_step:
        return delta
    return delta / norm_delta * max_step


class PickPlaceController:
    def __init__(self, env, ranges, tries=0):
        self._env = env
        self._g_tol = 5e-2 ** (tries + 1)
        self.ranges = ranges
        self.reset()

    def _calculate_quat(self, angle):
        if "Sawyer" in self._env.robot_names:
            new_rot = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
            return Quaternion(matrix=self._base_rot.dot(new_rot))
        return self._base_quat

    def reset(self):
        self._object_name = self._env.objects[self._env.object_id].name
        # TODO this line violates abstraction barriers but so does the reference implementation in robosuite
        self._jpos_getter = lambda: np.array(self._env._joint_positions)
        self._clearance = 0.03 if 'milk' not in self._object_name else -0.01

        if "Sawyer" in self._env.robot_names:
            self._obs_name = 'eef_pos'
            self._default_speed = 0.13
            self._final_thresh = 1e-2
            self._base_rot = np.array([[1, 0, 0.], [0, -1, 0.], [0., 0., -1.]])
            self._base_quat = Quaternion(matrix=self._base_rot)
        elif "Panda" in self._env.robot_names:
            self._obs_name = 'eef_pos'
            self._default_speed = 0.13
            self._final_thresh = 6e-2
            self._base_rot = np.array([[1, 0, 0.], [0, -1, 0.], [0., 0., -1.]])
            self._base_quat = Quaternion(matrix=self._base_rot)
        else:
            raise NotImplementedError

        self._t = 0
        self._intermediate_reached = False
        self._hover_delta = 0.25


    def _object_in_hand(self, obs):
        dist_panda = {'milk': 0.018, 'can': 0.018, 'cereal': 0.018, 'bread': 0.018}
        dist_sawyer = {'milk': 0.018, 'can': 0.018, 'cereal': 0.018, 'bread': 0.018}
        if "Panda" in self._env.robot_names:
            dist = dist_panda
        else:
            dist = dist_sawyer

        if np.linalg.norm(obs['{}_pos'.format(self._object_name)] - obs[self._obs_name]) < dist[self._object_name]:
            return True
        return False

    def _get_target_pose(self, delta_pos, base_pos, quat, max_step=None):
        if max_step is None:
            max_step = self._default_speed

        delta_pos = _clip_delta(delta_pos, max_step)

        if self.ranges.shape[0] == 7:
            aa = np.concatenate(([quat.angle / np.pi], quat.axis))
            if aa[0] < 0:
                aa[0] += 1
        else:
            quat = np.array([quat.x, quat.y, quat.z, quat.w])
            aa = quat2axisangle(quat)
        # absolute in world frame
        return normalize_action(np.concatenate((delta_pos + base_pos, aa)), self.ranges)

    def act(self, obs):
        self._target_loc = np.array(self._env.sim.data.body_xpos[self._env.bin_bodyid]) + [0, 0, 0.3]
        status = 'start'
        if self._t == 0:
            self._start_grasp = -1
            self._finish_grasp = False
            try:
                y = -(obs['{}_pos'.format(self._object_name)][1] - obs[self._obs_name][1])
                x = obs['{}_pos'.format(self._object_name)][0] - obs[self._obs_name][0]
            except:
                import pdb;
                pdb.set_trace()
            angle = np.arctan2(y, x) - np.pi / 3 if 'cereal' in self._object_name else np.arctan2(y, x)
            self._target_quat = self._calculate_quat(angle)

        if self._start_grasp < 0 and self._t < 15:
            if np.linalg.norm(obs['{}_pos'.format(self._object_name)] - obs[self._obs_name] + [0, 0,
                                                                                               self._hover_delta]) < self._g_tol or self._t == 14:
                self._start_grasp = self._t

            quat_t = Quaternion.slerp(self._base_quat, self._target_quat, min(1, float(self._t) / 5))
            eef_pose = self._get_target_pose(
                obs['{}_pos'.format(self._object_name)] - obs[self._obs_name] + [0, 0, self._hover_delta],
                obs['eef_pos'], quat_t)
            action = np.concatenate((eef_pose, [-1]))
            status = 'prepare_grasp'

        elif self._t < self._start_grasp + 30 and not self._finish_grasp:
            if not self._object_in_hand(obs):
                eef_pose = self._get_target_pose(
                    obs['{}_pos'.format(self._object_name)] - obs[self._obs_name] - [0, 0, self._clearance],
                    obs['eef_pos'], self._target_quat)
                action = np.concatenate((eef_pose, [-1]))
                self.object_pos = obs['{}_pos'.format(self._object_name)]
                status = 'reaching_obj'
            else:
                eef_pose = self._get_target_pose(self.object_pos - obs[self._obs_name] + [0, 0, self._hover_delta],
                                                 obs['eef_pos'], self._target_quat)
                action = np.concatenate((eef_pose, [1]))
                if np.linalg.norm(self.object_pos - obs[self._obs_name] + [0, 0, self._hover_delta]) < self._g_tol:
                    self._finish_grasp = True
                status = 'obj_in_hand'

        elif np.linalg.norm(
                self._target_loc - obs[self._obs_name]) > self._final_thresh and not self._intermediate_reached:
            target = self._target_loc
            eef_pose = self._get_target_pose(target - obs[self._obs_name], obs['eef_pos'], self._target_quat)
            action = np.concatenate((eef_pose, [1]))
            status = 'moving'
        else:
            self._intermediate_reached = True
            if np.linalg.norm(self._target_loc - [0, 0, 0.12] - obs[self._obs_name]) > self._final_thresh:
                target = self._target_loc - [0, 0, 0.12]
                eef_pose = self._get_target_pose(target - obs[self._obs_name], obs['eef_pos'], self._target_quat)
                action = np.concatenate((eef_pose, [1]))
            else:
                eef_pose = self._get_target_pose(np.zeros(3), obs['eef_pos'], self._target_quat)
                action = np.concatenate((eef_pose, [-1]))
            status = 'placing'

        self._t += 1
        return action, status


def get_expert_trajectory(env_type, controller_type, renderer=False, camera_obs=True, task=None, ret_env=False,
                          seed=None, env_seed=None, depth=False, heights=100, widths=200, gpu_id=0, **kwargs):
    assert 'gpu' in str(mujoco_py.cymj), 'Make sure to render with GPU to make eval faster'
    # reassign the gpu id
    visible_ids = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    gpu_id = int(visible_ids[gpu_id])

    seed = seed if seed is not None else random.getrandbits(32)
    env_seed = seed if env_seed is None else env_seed
    seed_offset = sum([int(a) for a in bytes(env_type, 'ascii')])
    np.random.seed(env_seed)
    if 'Sawyer' in env_type:
        action_ranges = np.array([[-0.05, 0.25], [-0.45, 0.5], [0.82, 1.2], [-5, 5], [-5, 5], [-5, 5]])
    else:
        action_ranges = np.array([[-0.05, 0.25], [-0.45, 0.5], [0.82, 1.2], [0.85, 1.08], [-1, 1], [-1, 1], [-1, 1]])
    success, use_object = False, None
    if task is not None:
        assert 0 <= task <= 15, "task should be in [0, 15]"
    else:
        raise NotImplementedError

    if ret_env:
        while True:
            try:
                env = get_env(env_type, controller_configs=controller_type,
                              task_id=task, has_renderer=renderer, has_offscreen_renderer=camera_obs,
                              reward_shaping=False, use_camera_obs=camera_obs, camera_heights=heights, camera_widths=widths,
                              camera_depths=depth, ranges=action_ranges,
                              camera_names="agentview", render_gpu_device_id=gpu_id, **kwargs)
                break
            except RandomizationError:
                pass
        return env

    tries = 0
    while True:
        try:
            env = get_env(env_type, controller_configs=controller_type,
                          task_id=task, has_renderer=renderer, has_offscreen_renderer=camera_obs,
                          reward_shaping=False, use_camera_obs=camera_obs, camera_heights=heights,
                          camera_widths=widths, camera_depths=depth, ranges=action_ranges,
                          camera_names="agentview",
                          render_gpu_device_id=gpu_id, **kwargs)
            break
        except RandomizationError:
            pass
    while not success:
        controller = PickPlaceController(env.env, tries=tries, ranges=action_ranges)
        np.random.seed(seed + int(tries) + seed_offset)
        while True:
            try:
                obs = env.reset()
                break
            except RandomizationError:
                pass
        mj_state = env.sim.get_state().flatten()
        sim_xml = env.model.get_xml()
        traj = Trajectory(sim_xml)

        env.reset_from_xml_string(sim_xml)
        env.sim.reset()
        env.sim.set_state_from_flattened(mj_state)
        env.sim.forward()
        use_object = env.object_id
        traj.append(obs, raw_state=mj_state, info={'status': 'start'})
        for t in range(int(env.horizon // 10)):
            action, status = controller.act(obs)
            obs, reward, done, info = env.step(action)
            assert 'status' not in info.keys(), "Don't overwrite information returned from environment. "
            info['status'] = status
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
    del controller
    del env
    return traj


if __name__ == '__main__':
    config = load_controller_config(default_controller='IK_POSE')
    for i in range(8, 16):
        traj = get_expert_trajectory('SawyerPickPlaceDistractor', config, renderer=True, camera_obs=False, task=i,
                                 render_camera='frontview')
