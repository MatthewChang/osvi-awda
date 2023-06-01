import numpy as np
import numpy as np
import numpy as np
import numpy as np
import re
import torch
import argparse
from envs.gripper_env import GripperEnv
from pyutil import *
import torch.utils.data
import argparse
import os
from dblog import DbLog
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from tqdm import tqdm
from einops import rearrange, repeat
import functools
import pickle
from hem.models.inverse_module import InverseImitation
from hem.datasets import get_dataset
from envs.metaworld_env import make_metaworld_single, ALL_ENVS
from hem.datasets.teacher_dataset import TeacherDemonstrations
from hem.datasets.agent_dataset import AgentDemonstrations
from hem.datasets.agent_teacher_dataset import AgentTeacherDataset
from policies.waypoints_policy import WaypointsPolicy
from utils.projection_utils import image_coords_to_3d
from utils.projection_utils import image_point_to_pixels, pixels_to_image_point, embed_mat,embed_matrix_square, compute_crop_adjustment
from hem.robosuite.controllers.expert_pick_place import get_expert_trajectory
import cv2
import pathlib

def sample_cont(dcfg,teacher_motion):
    dcfg = dcfg.copy()
    dcfg['test_tasks'] = [teacher_motion]
    dataset = AgentTeacherDataset(**dcfg,mode='test')
    ind = random.choice(np.arange(len(dataset)))
    if dcfg.get('mosaic'):
        task_string = pathlib.Path(dataset[ind][0]['fname']).parts[-2]
        subtask = int(re.match('task_(\d+)',task_string)[1])
    else:
        subtask = None
    return torch.tensor(dataset[ind][0]['video']), subtask

def test_transformer_all(config,**kwargs):
    _,nums = sorted_file_match(f"{config.logdir}/models",r'model_(\d+).torch')
    for n in nums:
        test_transformer(config,n,**kwargs)

def make_metaworld_env(task,cfg=None,seed=0):
    # not the most robust way to get the right depth
    # depth = 'depth' in cfg['dataset']['agent_dir']
    depth = False
    env = make_metaworld_single(task,random_init=True,seed=seed,rgb=(not depth),depth_only=depth,object_state=False,handcam=True)
    # force framestack 1 for now
    env = DictFrameStack(env,1)
    return env

def make_env(task,cfg=None,sub_task=None,seed=None):
    if seed is None:
        seed = random.randint(0,100000)
    if task in ALL_ENVS:
        return make_metaworld_env(task,cfg,seed)
    elif task == 'ost' or task in list(range(16)):
        return DictFrameStack(GripperEnv(task,rgb=True,handcam=True,seed=seed),1)
        # return get_expert_trajectory('PandaPickPlaceDistractor', task=sub_task, ret_env=True, seed=seed,hand_depth=True)
    else:
        return DictFrameStack(make_mosaic_env(task,sawyer=False,task=sub_task,handcam=True),1)

import pickle as pkl
from hem.datasets.agent_dataset import INVERSE_MATS
# PROJECTION_MATRICES = pkl.load(open('utils/transformation_mats_square.pkl','rb'))
# INVERSE_MATS = {key:np.linalg.inv(mat)for key,mat in PROJECTION_MATRICES.items()}
from envs.metaworld_env import ALL_ENVS
def get_pix_to_3d_projection(mot):
    if mot in ALL_ENVS:
        return INVERSE_MATS['metaworld']
    elif mot in list(range(16)):
        # crop_adjust = compute_crop_adjustment([100,0,0,0], [240,320])
        # return INVERSE_MATS['ost'] @ np.linalg.inv(crop_adjust)
        return INVERSE_MATS['ost']
    else:
        return INVERSE_MATS[mot] 

def test_transformer(logdir,config,num,writer=None,write_images=False,write_attention=False,random_policy=False,num_waypoints=None,instances=10,eval_only=False,motions=None,num_envs=40,dataset_number=0):
    if num_waypoints is None:
        num_waypoints = config['policy'].get('waypoints',0)
    print(f"loading model {num}")
    is_daml = 'maml_lr' in config['policy']
    if is_daml:
        import learn2learn as l2l
        from hem.models.baseline_module import DAMLNetwork
        model = DAMLNetwork(**config['policy'])
        model = l2l.algorithms.MAML(model, lr=config['policy']['maml_lr'], first_order=config['policy']['first_order'], allow_unused=True)
    else:
        model = InverseImitation(**config['policy'])
    model_path = os.path.join(logdir,f'model_save-{num}.pt')
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')).state_dict())
    model = model.eval().cuda()
    # model = config.build_model().cuda()
    # model.load_state_dict(torch.load(os.path.join(config.logdir,'models',f'model_{num}.torch'))) #type: ignore
    if dataset_number == 0:
        dataset_config = config['dataset']
    else:
        dataset_config = config['aux_datasets'][dataset_number-1]
        dataset_config = {**config['dataset'],**dataset_config}
    if dataset_number > 0:
        db_file = f'{logdir}/log{dataset_number}.db'
    else:
        db_file = f'{logdir}/log.db'
    dblog = DbLog(db_file)
    dataset_class = get_dataset(dataset_config.pop('type'))
    dataset = dataset_class(**dataset_config, mode='test')
    task_reses = []
    if eval_only:
        motions = dataset.test_tasks
    elif motions is None:
        motions = dataset.train_tasks + dataset.test_tasks
    num_held_out_tasks = len(dataset.test_tasks)
    all_mots = repeat(np.array(motions),'n -> n b', b = instances).ravel()
    splits = np.array_split(all_mots,math.ceil(len(all_mots)/num_envs))
    # splits = np.array_split(all_mots,math.ceil(len(all_mots)/100))
    counts = {}
    mosaic = dataset_config.get('mosaic',False)
    metaworld = dataset_config.get('metaworld',False)
    is_ost = not mosaic and not metaworld
    for mots in tqdm(splits):
        contexts,subtasks = zip(*[sample_cont(dataset_config,mot) for mot in mots])
        contexts = torch.stack(contexts).cuda()
        envs = SubprocVecEnv([functools.partial(make_env,m,config,st) for m,st in zip(mots,subtasks)],'spawn')
        projection_mats = torch.tensor(np.stack([get_pix_to_3d_projection(mot) for mot in mots])).cuda().float()
        # if config.args.novar or config.args.novar_val:
            # for i in range(envs.num_envs):
                # envs.env_method('set_task',tasks[i],indices=[i])
        # if config.args.no_context:
            # contexts=None
        # ROLLOUTS
        states = []
        ims = []
        actions = []
        rewards = []
        infos = []
        masks = []
        state = envs.reset()
        # compute waypoints
        # if config.args.start_only:
        states.append(state)
        render = lambda: np.stack(envs.get_images())

        start_im = state['img'][...,:3]
        resized = np.stack([dataset._agent_dataset._crop_and_resize(x[0]) for x in start_im])
        start_im,transform_stats = dataset._agent_dataset.randomize_frames(resized,ret_stats=True)
        # start_im,transform_stats = train_dataset._agent_dataset.randomize_frames(resized,ret_stats=True)

        im_shape = start_im.shape[-3:-1]
        crop_adjust = compute_crop_adjustment(dataset._agent_dataset._crop, im_shape)
        trans_mat = torch.tensor(np.linalg.inv(crop_adjust)).cuda().float()
        projection_mats = torch.stack([mat @ trans_mat for mat in projection_mats])

        start_im = to_channel_first(torch.tensor(start_im[:,None]))
        start_im = start_im.cuda()
        start_im = repeat(start_im,'b 1 c r col->b 2 c r col')
        policies = []
        if write_images:
            ims.append(render())
        if config.get('waypoints',False):
            with torch.no_grad():
                start_poses = state['state'][:,-1,:4] #type: ignore
                flatstate = rearrange(state['state'],'b t d -> b (t d)') #type: ignore
                head_label = dataset_config.get('head_label',0)
                head_label = torch.ones((start_im.shape[0],)).cuda()*head_label
                out = model(flatstate, start_im, contexts, ents=head_label,ret_dist=False)
                # Timing
                # import time
                # ct = time.time()
                # out = model(flatstate[:1], start_im[:1], contexts[:1], ents=head_label[:1],ret_dist=False)
                # elapsed_time = time.time() - ct
                # print('inference_time for 1 env', elapsed_time)
                if config.get('image_waypoints'):
                    waypoints = image_coords_to_3d(out['waypoints'],projection_mats)
                    waypoints[...,3:] = out['waypoints'][...,3:]
                    world_waypoints = waypoints.cpu().numpy()
                else:
                    waypoints = out['waypoints'].cpu().numpy()
                    shiftby = start_poses[:,None,:]
                    shiftby[:,:,-1] = 0
                    world_waypoints = waypoints + shiftby
                # waypoints = model(torch.tensor(flatstate).cuda().float(),contexts,ims=start_im).mean.cpu().numpy()
                # world_waypoints = waypoints[:,:,:4] + start_poses[:,None,:]
                if config['policy'].get('sub_waypoints',False):
                    start = num_waypoints*(num_waypoints-1)//2
                else:
                    start = 0
                inter_waypoints = world_waypoints[:,start:start+num_waypoints]
                for i,wp in enumerate(inter_waypoints):
                    envs.env_method('set_waypoints',wp,indices=[i])
                policies = [WaypointsPolicy(x,panda=(is_ost or mosaic),mosaic=mosaic,verbose=False) for x in inter_waypoints]
                # import pdb; pdb.set_trace()
                # print("SAVING")
                # with open('objs/bball_waypoints.pkl', 'wb') as fil: pickle.dump(policies, fil)
                '''
                imstack = np.stack(envs.get_images())
                from matplotlib import pyplot as plt 
                plt.imsave('vis/test0.png',rearrange(imstack,'b r c ch -> (b r) c ch'))
                envs.get_attr('waypoints')[1]
                inter_waypoints[1]
                with open('objs/bball_waypoints.pkl', 'wb') as fil: pickle.dump(policies, fil)
                '''
        done = None
        total_rew = 0
        info = None
        images = []
        hand_cam_images = []
        finished = None
        while finished is None or not finished.all():
            if config.get('waypoints',False):
                actions = []
                for st,im,policy in zip(state['state'],state['img'],policies): #type: ignore
                    assert st.shape[0] == 1 and im.shape[0] == 1
                    action = policy.act({'state':st[0],'img':im[0,:,:,-1]})[0]
                    actions.append(action)
                action = np.stack(actions)
                # if config.args.start_only:
                # curway = np.take_along_axis(inter_waypoints,current_waypoints[:,None,None],axis=1).squeeze(1) #type: ignore
                # action = waypoints_to_actions(state['state'],curway) #type: ignore
                # else:
                    # raise Exception(f'not implemented')
                    # print("using start only")
                    # action = waypoints_to_actions(state['state'],inter_waypoints) #type: ignore
            elif random_policy:
                action =  np.stack([envs.action_space.sample() for _ in range(len(mots))])
            else:
                if len(images) >= dataset_config['T_pair']:
                    images = images[1:]
                    hand_cam_images = hand_cam_images[1:]

                inims = state['img'][:,0,:,:,:3]
                resized = np.stack([dataset._agent_dataset._crop_and_resize(x) for x in inims])
                images.append(resized)

                inims = state['img'][:,0,:,:,3:]
                resized = np.stack([dataset._agent_dataset._crop_and_resize(x) for x in inims])
                hand_cam_images.append(resized)

                randomized = np.stack([dataset._agent_dataset.randomize_frames(ims) for ims in np.stack(images,axis=1)])
                randomized = to_channel_first(randomized)


                if 'vis' in config['policy'] and config['policy']['vis'].get('hand_cam',False):
                    adj_handcam = np.stack([dataset._agent_dataset.randomize_frames(ims) for ims in np.stack(hand_cam_images,axis=1)])
                    randomized = np.concatenate((randomized,to_channel_first(adj_handcam)),axis=-3)

                # from einops import rearrange, reduce, repeat
                # from matplotlib import pyplot as plt 
                # nobs = rearrange(randomized[:10],'b t h w c-> (b h) (t w) c')
                if is_daml:
                    inner_iters = config.get('inner_iters', 1)
                    batch_size = contexts.shape[0]
                    imbatch = torch.tensor(randomized).cuda()
                    action = []
                    for sid in range(batch_size):
                        learner = model.clone()
                        for _ in range(inner_iters):
                            learner.adapt(learner(None, contexts[sid], learned_loss=True)['learned_loss'])
                        out = learner(state['state'], imbatch[sid], ret_dist=True)
                        action.append(out['action_dist'].sample()[-1].cpu().detach().numpy())
                    action = np.stack(action)
                else:
                    with torch.no_grad():
                        out = model(state['state'], torch.tensor(randomized).cuda(), contexts, ret_dist=True)
                        action = out['bc_distrib'].sample()[:,-1].cpu().numpy()
            if mosaic:
                rots = repeat(default_rot,'d -> b d', b = mots.shape[0])
                rots[mots == 'button'] = button_rot
                action = np.concatenate((action[:,:3],rots,action[:,-1:]),axis=-1)
                # close for button
                rots[mots == 'button',-1] = 1
            state,reward,done,info = envs.step(action)
            finished = np.logical_or(finished,done) if finished is not None else done
            # if config.get('waypoints',False):
                # poses = states_to_poses(state['state']) #type: ignore
                # current_waypoints = current_waypoints + (np.linalg.norm(poses-curway,axis=1)<0.01).astype(int) #type: ignore
                # current_waypoints = np.clip(current_waypoints,0,num_waypoints-1)
            if write_images:
                ims.append(render())
            states += [state]
            actions += [action]
            rewards += [reward]
            infos += [info]
            total_rew += reward
            masks.append(finished.copy())
        if write_images:
            ims = np.array(ims)
        states,action,rewards,infos,masks= np.array(states),np.array(actions),np.array(rewards),infos, np.array(masks)
        # END ROLLOUT
        rew_tot = rewards.sum(axis=0)
        term_indices = [np.where(mask)[0].min() for mask in masks.T]
        successes = [int(infos[t][i]['is_success']) for i,t in enumerate(term_indices)]
        # successes = [int(i['is_success']) for i in infos[-1]]
        
        if write_images:
            for i,mot in enumerate(mots):
                counts[mot] = counts.get(mot,0) + 1
                di = logdir+f'/evals/{num}/{mot}'
                pathlib.Path(di).mkdir(parents=True, exist_ok=True)
                fps = 45 if metaworld else 10
                mask = np.concatenate(([False],masks[:,i]))
                to_render = ims[~mask,i]
                if successes[i] and 'success_frame' in infos[term_indices[i]][i]:
                    last_frame = infos[term_indices[i]][i]['success_frame']['image']
                    to_render = np.concatenate((to_render,last_frame[None]))
                fname = f'{di}/{counts[mot]}-{successes[i]}.mp4'
                imageio.mimwrite(fname,to_render,fps=fps) #type: ignore
                print(f'writing {fname}')
                context_flat = rearrange(contexts[i],'t c w h -> (t w) h c')
                context_flat = (context_flat-context_flat.min())/(context_flat.max()-context_flat.min())*255
                imageio.imwrite(f'{di}/{counts[mot]}_context.jpg',context_flat.cpu().numpy().astype(np.uint8))
        task_reses.append(np.stack((successes,rew_tot),axis=1))
    res = rearrange(np.concatenate(task_reses),'(n b) d -> n b d',b=instances)
    num = int(num)
    if eval_only:
        print("SR out", res[:,:,0].mean())
        print("RW out", res[:,:,1].mean())
    for mot,results in zip(motions,res): 
        writer.add_scalar(f"eval/{mot}/success_rate_out_dist", results[:,0].mean(), num)
        writer.add_scalar(f"eval/{mot}/mean_reward_out_dist", results[:,1].mean(), num)
        dblog.log(f"eval/{mot}/mean_reward_out_dist", results[:,1].mean(), num)
        dblog.log(f"eval/{mot}/success_rate_out_dist", results[:,0].mean(), num)
    if eval_only:
        print("SR out", res[:,:,0].mean())
        print("RW out", res[:,:,1].mean())
        writer.add_scalar("eval/success_rate_out_dist", res[:,:,0].mean(), num)
        writer.add_scalar("eval/mean_reward_out_dist", res[:,:,1].mean(), num)
        dblog.log("eval/mean_reward_out_dist", res[:,:,1].mean(), num)
        dblog.log("eval/success_rate_out_dist", res[:,:,0].mean(), num)
        return
    if random_policy:
        import pdb; pdb.set_trace()
    if writer is not None:
        writer.add_scalar("eval/success_rate", res[:,:,0].mean(), num)
    dblog.log("eval/success_rate", res[:,:,0].mean(), num)
    if writer is not None:
        writer.add_scalar("eval/success_rate_out_dist", res[-num_held_out_tasks:,:,0].mean(), num)
        writer.add_scalar("eval/mean_reward_out_dist", res[-num_held_out_tasks:,:,1].mean(), num)
        dblog.log("eval/mean_reward_out_dist", res[-num_held_out_tasks:,:,1].mean(), num)
        dblog.log("eval/success_rate_out_dist", res[-num_held_out_tasks:,:,0].mean(), num)
    print("SR out", res[-num_held_out_tasks:,:,0].mean())
    indistres = res[:-num_held_out_tasks]
    print("SR in", indistres[:,:,0].mean())
    if writer is not None:
        writer.add_scalar("eval/success_rate_in_dist", indistres[:,:,0].mean(), num)
        writer.add_scalar("eval/mean_reward_in_dist", indistres[:,:,1].mean(), num)
        dblog.log("eval/success_rate_in_dist", indistres[:,:,0].mean(), num)
        dblog.log("eval/mean_reward_in_dist", indistres[:,:,1].mean(), num)
        writer.flush()
    # writer.close()
    # dblog.close()

def states_to_poses(states):
    if len(states.shape) == 3:
        return states[...,-1,:3]
    else:
        return states[...,:3]

def waypoints_to_actions(states,waypoints):
    poses = states_to_poses(states)
    diffs = waypoints - poses 
    actions = np.clip(diffs*10,-1,1)
    grips = np.ones((states.shape[0],1))*-1
    return np.concatenate((actions,grips),axis=-1)

