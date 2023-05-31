from torch.utils.data import Dataset
from einops import repeat
import cv2
import random
import os
import torch
from hem.datasets.util import resize, crop, randomize_video
import random
import numpy as np
import io
import tqdm
from hem.datasets import get_files, load_traj
from hem.datasets.util import split_files
import pickle as pkl
from envs.metaworld_env_data import ALL_ENVS
from envs.mosaic_env_data import TASK_ENV_MAP
from utils.projection_utils import image_point_to_pixels, pixels_to_image_point, embed_mat,embed_matrix_square, compute_crop_adjustment

bcz_correction = np.array([[1./320, 0, -1], [0., -1./256, 1], [0., 0., 1.]])
def traj_to_base_matrix(traj):
    if 'world_to_image_transform' in traj.get(0)['obs']:
        return np.linalg.inv(traj.get(0)['obs']['world_to_image_transform'])
    if 'cam_mat' in traj.get(0)['obs']:
        return np.linalg.inv(embed_mat(bcz_correction @ traj.get(0)['obs']['cam_mat']))
    if traj.setting_name in ALL_ENVS:
        return INVERSE_MATS['metaworld']
    elif traj.setting_name in TASK_ENV_MAP.keys():
        return INVERSE_MATS[traj.setting_name]
    else:
        return INVERSE_MATS['ost']


def adjust_augmentations(stats,size):
    # account for random crops and translations
    random_crop_adjust = compute_crop_adjustment(stats['crop'],size)
    random_trans_adjust = np.eye(4)
    # 'trans' output is in pixels, [x,y] format, so scale by 2*[c,r] and flip y
    trans = 2*stats['trans']/np.flip(size)
    trans[1] = -trans[1]
    random_trans_adjust[:3,2] = embed_mat(trans,3)
    return np.linalg.inv(random_crop_adjust) @ np.linalg.inv(random_trans_adjust) @embed_mat(stats['flip'])

PROJECTION_MATRICES = pkl.load(open('utils/transformation_mats_square.pkl','rb'))
INVERSE_MATS = {key:np.linalg.inv(mat)for key,mat in PROJECTION_MATRICES.items()}
INVERSE_MATS['dev'] = np.linalg.inv(pkl.load(open('utils/transformation_mat_ost.pkl','rb')))
INVERSE_MATS['ost'] = INVERSE_MATS['dev']
INVERSE_MATS['.'] = INVERSE_MATS['dev']

class AgentDemonstrations(Dataset):
    def __init__(self, root_dir=None, files=None, height=224, width=224, depth=False, normalize=True, crop=None, randomize_vid_frames=False, T_context=15, extra_samp_bound=0,
                 T_pair=0, freq=1, append_s0=False, mode='train', split=[0.9, 0.1], state_spec=None, action_spec=None, sample_sides=False, min_frame=0, cache=False, random_targets=False,
                 color_jitter=None, rand_crop=None, rand_rotate=None, is_rad=False, rand_translate=None, rand_gray=None, rep_buffer=0, target_vid=False, reduce_bits=False, aux_pose=False,waypoints=False,no_context_jitter=False,grasp=True,rand_flip=False,interpolate_gaps=False,high_ent=False,head_label=0,raw_images=False,start_samp_rate=0):
        assert mode in ['train', 'val'], "mode should be train or val!"
        assert T_context >= 2 or T_pair > 0, "Must return (s,a) pairs or context!"

        self.raw_images = raw_images
        self._rand_flip = rand_flip
        self.waypoints = waypoints
        self.no_context_jitter = no_context_jitter
        self.interpolate_gaps = interpolate_gaps
        self.high_ent = high_ent
        self.head_label=head_label
        self.start_samp_rate = start_samp_rate
        if high_ent and head_label == 0:
            raise Exception(f'bad label')
        if files is None and rep_buffer:
            all_files = []
            for f in range(rep_buffer):
                all_files.extend(pkl.load(open(os.path.expanduser(root_dir.format(f)), 'rb')))
            order = split_files(len(all_files), split, mode)
            files = [all_files[o] for o in order]
        elif files is None:
            all_files = get_files(root_dir)
            order = split_files(len(all_files), split, mode)
            files = [all_files[o] for o in order]

        self._trajs = files
        if cache:
            for i in tqdm.tqdm(range(len(self._trajs))):
                if isinstance(self._trajs[i], str):
                    self._trajs[i] = load_traj(self._trajs[i])

        self._im_dims = (width, height)
        self._randomize_vid_frames = randomize_vid_frames
        self._crop = tuple(crop) if crop is not None else (0, 0, 0, 0)
        self._depth = depth
        self._normalize = normalize
        self._T_context = T_context
        self._T_pair = T_pair
        self._freq = freq
        state_spec = tuple(state_spec) if state_spec is not None else ('ee_aa', 'ee_vel', 'joint_pos', 'joint_vel', 'gripper_qpos', 'object_detected')
        action_spec = tuple(action_spec) if action_spec is not None else ('action',)
        self._state_action_spec = (state_spec, action_spec)
        self._color_jitter = color_jitter
        self._rand_crop = rand_crop
        self._rand_rot = rand_rotate if rand_rotate is not None else 0
        if not is_rad:
            self._rand_rot = np.radians(self._rand_rot)
        self._rand_trans = np.array(rand_translate if rand_translate is not None else [0, 0])
        self._rand_gray = rand_gray
        self._normalize = normalize
        self._append_s0 = append_s0
        self._sample_sides = sample_sides
        self._target_vid = target_vid
        self._reduce_bits = reduce_bits
        self._min_frame = min_frame
        self._extra_samp_bound = extra_samp_bound
        self._random_targets = random_targets
        self._aux_pose = aux_pose
        self.grasp = grasp

    def __len__(self):
        return len(self._trajs)
    
    def __getitem__(self, index,force_flip = None):
        if torch.is_tensor(index):
            index = index.tolist()
        assert 0 <= index < len(self._trajs), "invalid index!"
        return self.proc_traj(self.get_traj(index),force_flip)
    
    def get_traj(self, index):
        if isinstance(self._trajs[index], str):
            return load_traj(self._trajs[index])
        return self._trajs[index]

    def proc_traj(self, traj,force_flip=None):
        context_frames = []
        if self._T_context:
            context_frames = self._make_context(traj,force_flip=force_flip)

        if self._T_pair == 0:
            return {}, context_frames
        return self._get_pairs(traj,force_flip=force_flip), context_frames

    def _make_context(self, traj,force_flip=None):
        clip = lambda x : int(max(0, min(x, len(traj) - 1)))
        per_bracket = max(len(traj) / self._T_context, 1)
        def _make_frame(n):
            obs = traj.get(n)['obs']
            img = self._crop_and_resize(obs['image'])
            if self._depth:
                img = np.concatenate((img, self._crop_and_resize(obs['depth'][:,:,None])), -1)
            return img[None]

        frames = []
        if self.no_context_jitter:
            teacher_im_inds = np.linspace(0,len(traj)-1,num=self._T_context,endpoint=True,dtype=int)
            frames = [_make_frame(i) for i in teacher_im_inds]
        else:
            for i in range(self._T_context):
                n = clip(np.random.randint(int(i * per_bracket), int((i + 1) * per_bracket)))
                if self._sample_sides and i == self._T_context - 1:
                    n = len(traj) - 1
                elif self._sample_sides and i == 0:
                    n = self._min_frame
                frames.append(_make_frame(n))
        frames = np.concatenate(frames, 0)
        frames,stats = randomize_video(frames, self._color_jitter, self._rand_gray, self._rand_crop, self._rand_rot, self._rand_trans, self._normalize,rand_flip=self._rand_flip,force_flip=force_flip)
        projection = traj_to_base_matrix(traj)
        size = frames.shape[-3:-1]
        crop_adjust = compute_crop_adjustment(self._crop, size)
        projection = projection @ np.linalg.inv(crop_adjust)
        projection = projection @ adjust_augmentations(stats,size)
        return {'video': np.transpose(frames, (0, 3, 1, 2)), 'projection_matrix': projection,'fname':traj.fname}

    def _get_pairs(self, traj, end=None,force_flip=None):
        def _get_tensor(k, t):
            if k == 'action':
                return t['action']
            elif k == 'grip_action':
                return [t['action'][-1]]

            o = t['obs']
            if k == 'ee_aa' and 'ee_aa' not in o:
                ee, axis_angle = o['ee_pos'][:3], o['axis_angle']
                if axis_angle[0] < 0:
                    axis_angle[0] += 2
                o = np.concatenate((ee, axis_angle)).astype(np.float32)
            else:
                o = o[k]
            return o
        
        state_keys, action_keys = self._state_action_spec
        ret_dict = {'images': [], 'states': [], 'actions': []}
        if self.raw_images:
            ret_dict['raw_images'] = []
        if self._depth:
            ret_dict['depth'] = []
        if len(traj) == 0:
            import pdb; pdb.set_trace()

        traj_elements = [x for x in traj]
        has_eef_point = 'eef_point' in traj_elements[0]['obs']
        if has_eef_point:
            ret_dict['points'] = []

        end = len(traj) if end is None else end
        start = np.random.randint(0, max(1, end - self._T_pair * self._freq))
        if np.random.uniform() < self._extra_samp_bound:
            start = 0 if np.random.uniform() < 0.5 else max(1, end - self._T_pair * self._freq) - 1
        if np.random.uniform() < self.start_samp_rate: start = 0
        if self.waypoints:
            start = 0
        chosen_t = [j * self._freq + start for j in range(self._T_pair + 1)]
        if self._append_s0:
            chosen_t = [0] + chosen_t

        for j, t in enumerate(chosen_t):
            if len(traj) < 2:
                print(traj.fname, t, len(traj))
                assert False, traj.fname
            t = traj_elements[t]
            if self._depth:
                depth_img = self._crop_and_resize(t['obs']['depth']).transpose((2, 0, 1))[None]
                ret_dict['depth'].append(depth_img)
            image = t['obs']['image']
            if self.raw_images:
                ret_dict['raw_images'].append(image[None])
            ret_dict['images'].append(self._crop_and_resize(image)[None])
            if 'hand_cam' in t['obs']:
                if 'hand_cam' not in ret_dict: ret_dict['hand_cam'] = []
                him = repeat(t['obs']['hand_cam'],'h w -> h w 3')
                ret_dict['hand_cam'].append(self._crop_and_resize(him)[None])
            if has_eef_point:
                ret_dict['points'].append(np.array(self._adjust_points(t['obs']['eef_point'], image.shape[:2]))[None])
            state = []
            for k in state_keys:
                state.append(_get_tensor(k, t))
            if len(state) == 0: state = [[0]]
            ret_dict['states'].append(np.concatenate(state).astype(np.float32)[None])
            
            if j > 1 or (j==1 and not self._append_s0):
                action = []
                for k in action_keys:
                    action.append(_get_tensor(k, t))
                if len(action) == 0: action = [[0]]
                ret_dict['actions'].append(np.concatenate(action).astype(np.float32)[None])
        for k, v in ret_dict.items():
            ret_dict[k] = np.concatenate(v, 0).astype(np.float32)
        if self._target_vid:
            ret_dict['target_images'] = randomize_video(ret_dict['images'].copy(), normalize=False)[0].transpose((0, 3, 1, 2))
        if self._random_targets:
            raise Exception(f'not used')
            # ret_dict['transformed'] = [randomize_video([f], self._color_jitter, self._rand_gray, self._rand_crop, self._rand_rot, self._rand_trans, self._normalize) for f in ret_dict['images']]
            # ret_dict['transformed'] = np.concatenate(ret_dict['transformed'], 0).transpose(0, 3, 1, 2)

        ret_dict['images'],randomize_stats = self.randomize_frames(ret_dict['images'],ret_stats=True,force_flip=force_flip)
        # if self._randomize_vid_frames:
            # ret_dict['images'] = [randomize_video([f], self._color_jitter, self._rand_gray, self._rand_crop, self._rand_rot, self._rand_trans, self._normalize) for f in ret_dict['images']]
            # ret_dict['images'] = np.concatenate(ret_dict['images'], 0)
        # else:
            # ret_dict['images'] = randomize_video(ret_dict['images'], self._color_jitter, self._rand_gray, self._rand_crop, self._rand_rot, self._rand_trans, self._normalize)
        ret_dict['images'] = np.transpose(ret_dict['images'], (0, 3, 1, 2))
        if 'hand_cam' in ret_dict:
            ret_dict['hand_cam'] = np.transpose(ret_dict['hand_cam'], (0, 3, 1, 2))

        # flip points for baseline if needed:
        for ind,stats in enumerate(randomize_stats):
            flip = stats['flip']
            sh = ret_dict['images'][ind].shape[1:]
            if flip[1,1] == -1:
                ret_dict['points'][ind][0] = sh[0] - ret_dict['points'][ind][0] - 1
            if flip[0,0] == -1:
                ret_dict['points'][ind][1] = sh[1] - ret_dict['points'][ind][1] - 1

        if self._aux_pose:
            grip_close = np.array([traj_elements[i]['action'][-1] > 0 for i in range(1, len(traj))])
            grip_t = np.argmax(grip_close)
            drop_t = len(traj) - 1 - np.argmax(np.logical_not(grip_close)[::-1])
            aux_pose = [traj_elements[i]['obs']['ee_aa'][:3] for t in (grip_t, drop_t)]
            ret_dict['aux_pose'] = np.concatenate(aux_pose).astype(np.float32)

        out_inds = np.linspace(0,len(traj)-1,num=50,endpoint=True,dtype=int)#type: ignore

        ret_dict['setting_name'] = traj.setting_name
        if 'grasp' in traj_elements[0]['obs']:
            # for metaworld/bcz/mosaic
            grasp_frames = [traj_elements[i]['obs']['grasp'] for i in range(0,len(traj))]
        else:
            # for pick place env
            grasp_frames = [False] + [traj_elements[i]['action'][-1] > 0.01 for i in range(1,len(traj))]
        
        if self.interpolate_gaps:
            poses = np.stack([traj_elements[i]['obs']['ee_aa'][:3] for i in range(len(traj))])
            grasp_ints = np.array(grasp_frames).astype(int)
            poses_with_grasps = np.concatenate((poses,grasp_ints[:,None]),axis=-1)
            all_poses = []
            for i in range(len(poses_with_grasps)):
                if i > 0:
                    # check if eef position has moved more than 0.1 (ignoring grasp)
                    diff = np.linalg.norm(poses_with_grasps[i-1,:-1]-poses_with_grasps[i,:-1],axis=-1)
                    if diff > 0.1:
                        all_poses.append(np.linspace(poses_with_grasps[i-1],poses_with_grasps[i],10,endpoint=True))
                        continue
                all_poses.append(poses_with_grasps[i:i+1])
            all_poses = np.concatenate(all_poses,axis=0)
            out_inds = np.linspace(0,len(all_poses)-1,num=50,endpoint=True,dtype=int)#type: ignore
            poses = all_poses[out_inds,:-1]
            grasps = all_poses[out_inds,-1]
        else:
            out_inds = np.linspace(0,len(traj)-1,num=50,endpoint=True,dtype=int)#type: ignore
            poses = np.stack([traj_elements[i]['obs']['ee_aa'][:3] for i in out_inds])
            grasps = np.stack([grasp_frames[i] for i in out_inds]).astype(np.int32)
        # grasp scale is 0.2 this is tunable
        traj_points = np.concatenate((poses,grasps[:,None]*0.2),axis=-1)
        ret_dict['traj_points'] = traj_points
        ret_dict['start0'] = start == 0
        grasp_ind = np.where(grasps)[0]
        if len(grasp_ind) > 0 and self.grasp:
            ret_dict['grasp_point'] = poses[grasp_ind[0]]
        else:
            ret_dict['grasp_point'] = np.zeros((3,))

        ret_dict['projection_matrix'] = traj_to_base_matrix(traj)
        size = ret_dict['images'].shape[-2:]
        crop_adjust = compute_crop_adjustment(self._crop, size)
        ret_dict['projection_matrix'] = ret_dict['projection_matrix'] @ np.linalg.inv(crop_adjust)
        ret_dict['projection_matrix'] = ret_dict['projection_matrix'] @ adjust_augmentations(randomize_stats[0],size)
        ret_dict['high_ent'] = int(self.high_ent)
        ret_dict['head_label'] = int(self.head_label)
        return ret_dict

    def randomize_frames(self,frames,ret_stats = False,force_flip=None):
        if self._randomize_vid_frames:
            result = [randomize_video([f], self._color_jitter, self._rand_gray, self._rand_crop, self._rand_rot, self._rand_trans, self._normalize,rand_flip = self._rand_flip,force_flip=force_flip) for f in frames]
            result,stats = zip(*result)
            result = np.concatenate(result, 0)
        else:
            result,stats = randomize_video(frames, self._color_jitter, self._rand_gray, self._rand_crop, self._rand_rot, self._rand_trans, self._normalize,rand_flip = self._rand_flip,force_flip=force_flip)
            stats = [stats]
        if ret_stats:
            return result,stats
        else:
            return result
    
    def _crop_and_resize(self, img, normalize=False):
        return resize(crop(img, self._crop), self._im_dims, normalize, self._reduce_bits)
    
    def _adjust_points(self, points, frame_dims):
        h = np.clip(points[0] - self._crop[0], 0, frame_dims[0] - self._crop[1])
        w = np.clip(points[1] - self._crop[2], 0, frame_dims[1] - self._crop[3])
        h = float(h) / (frame_dims[0] - self._crop[0] - self._crop[1]) * self._im_dims[1]
        w = float(w) / (frame_dims[1] - self._crop[2] - self._crop[3]) * self._im_dims[0]
        return tuple([int(min(x, d - 1)) for x, d in zip([h, w], self._im_dims[::-1])])


if __name__ == '__main__':
    import time
    import imageio
    from torch.utils.data import DataLoader
    batch_size = 10
    ag = AgentDemonstrations('/dev/shm/mc48/metaworld_traj_eef_depth', normalize=False)
    loader = DataLoader(ag, batch_size = batch_size, num_workers=8)

    start = time.time()
    timings = []
    for pairs, context in loader:
        timings.append(time.time() - start)
        print(context.shape)

        if len(timings) > 1:
            break
        start = time.time()
    print('avg ex time', sum(timings) / len(timings) / batch_size)

    out = imageio.get_writer('out1.gif')
    for t in range(context.shape[1]):
        frame = [np.transpose(fr, (1, 2, 0)) for fr in context[:, t]]
        frame = np.concatenate(frame, 1)
        out.append_data(frame.astype(np.uint8))
    out.close()
