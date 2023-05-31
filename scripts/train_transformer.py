import torch
from hem.models.inverse_module import InverseImitation
from hem.models import Trainer
from hem.models.discrete_logistic import DiscreteMixLogistic
import numpy as np
import matplotlib.pyplot as plt
from hem.datasets.util import MEAN, STD
import cv2
from einops import rearrange, reduce, repeat
from soft_dtw_cuda import SoftDTW
from pyutil import set_seed, experiment_log

import os
# os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
# os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
# os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
# os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1
# torch.set_num_threads(1)

def compute_loss_trajectory(waypoints,traj_points,projection,num_points,gamma,kwargs,image_waypoints=False):
        cureepos = traj_points[:,0]
        future_poses = traj_points
        all_waypoints = waypoints
        device = waypoints.device
        if not kwargs.get('grasp',True):
            all_waypoints[...,-1] = 0
            cureepos[...,-1] = 0
            future_poses[...,-1] = 0
        point_sets = []
        transind = 0
        if image_waypoints:
            # convert normal image coords to homogenious coords by scaling by third component
            hom_im_coords = torch.cat((all_waypoints[:,:,:2]*all_waypoints[:,:,2:3],all_waypoints[:,:,2:3]),axis=-1)
            # add 1 to be homogenious for 4d conversion
            hom_im_4d = torch.cat((hom_im_coords, torch.ones(*hom_im_coords.shape[:2],1,device=device)),axis=-1)
            # do the matrix multiplication
            trans_waypoints = torch.einsum('bwh,bdh->bwd',hom_im_4d,projection.float())
            # replace the first 3d while keeping the higher dimensions
            all_waypoints = torch.cat((trans_waypoints[:,:,:3],all_waypoints[:,:,3:]),axis=-1)
        if kwargs.get('sub_waypoints',False):
            waypoint_counts = list(range(1,kwargs.get('waypoints')+1))
        else:
            waypoint_counts = [waypoints.shape[1]]
        for num_waypoints in waypoint_counts:
            if image_waypoints:
                prev_point = cureepos
            else:
                prev_point = torch.zeros_like(cureepos)
            inter_points = []
            for i in range(num_waypoints):
                waypoints = all_waypoints[:,transind]
                diffs = waypoints-prev_point
                # only include endpoint for last waypoint
                points = torch.tensor(np.linspace(0,1,num=num_points//num_waypoints,endpoint=(i==(num_waypoints-1))),device=device)
                inter_points_raw = torch.einsum('bd,t->btd',diffs,points)+prev_point.unsqueeze(1)
                # set grasp value to fixed to waypoint value, so the whole segment has same value
                inter_points_raw[...,-1] = waypoints[...,-1].unsqueeze(-1)
                inter_points.append(inter_points_raw)
                prev_point=waypoints
                transind += 1
            inter_points = torch.cat(inter_points,dim=1)
            point_sets.append(inter_points)
        point_sets = torch.stack(point_sets)
        # only shift if doing image waypoints, otherwise we're predicting image coordinates
        if not image_waypoints:
            # shift future poses to be relative to current ee pos, this makes predictions relative
            # NOTE: that since all of the grippers start open, this is shifting grasp predictions to be -1->0
            future_poses = future_poses - cureepos.unsqueeze(1)
        future_poses_batched = repeat(future_poses,'b t d -> (w b) t d',w=point_sets.shape[0])
        point_sets_batched = rearrange(point_sets,'w b t d -> (w b) t d')
        # higher gamma values results in overshooting if there is a cluster of states at the end
        # (i.e. the gripper sitting in the goal pos). The right thing might be to anneal this value
        # to 0 over training
        sdtw = SoftDTW(use_cuda=True, gamma=gamma)
        loss = sdtw(future_poses_batched,point_sets_batched)
        # logging
        log_items = {}
        log_items['loss_batch'] = rearrange(loss,'(w b) -> b w',w=point_sets.shape[0]).mean(axis=1)
        log_items['gamma'] = gamma
        losses = rearrange(loss,'(w b) -> w b', w=point_sets.shape[0]).mean(axis=1)
        for i,counts in enumerate(waypoint_counts):
            log_items[f'loss_w{counts-1}'] = losses[i].item()
        return loss.mean(axis=0), log_items

def forward(config,m, device, context, traj, append=True,val=False):
    pnt_weight = config.get('pnt_weight', 0.1)
    states, actions = traj['states'], traj['actions']
    images = traj['images']
    context_projection = context['projection_matrix']
    context = context['video']
    device = context.device
    traj_points = traj['traj_points']

    if 'hand_cam' in traj:
        images = torch.cat((images,traj['hand_cam']),axis=-3)
    if config.get('mixup',False):
        mix_rate = torch.tensor(np.random.uniform(0.3,1,(images.shape[0]))).to(device).float()
        shiftinds = np.concatenate((np.arange(1,images.shape[0]),[0]))
        batch_dot = lambda x,y: torch.einsum('btchw,b -> btchw',x,y)
        instance_first_images = repeat(images[:,0:1],'b 1 c h w -> b t c h w',t=images.shape[1])
        images = batch_dot(images,mix_rate)+batch_dot(instance_first_images[shiftinds],1-mix_rate)
        context_first_images = repeat(context[:,0:1],'b 1 c h w -> b t c h w',t=context.shape[1])
        context = batch_dot(context,mix_rate)+batch_dot(context_first_images[shiftinds],1-mix_rate)
    if config.get('orig_mixup',False):
        alpha = 1
        mix_rate = torch.tensor(np.random.beta(alpha, alpha,size=(images.shape[0]))).to(device).float()
        shiftinds = np.concatenate((np.arange(1,images.shape[0]),[0]))
        batch_dot = lambda x,y: torch.einsum('btchw,b -> btchw',x,y)
        images = batch_dot(images,mix_rate)+batch_dot(images[shiftinds],1-mix_rate)
        context = batch_dot(context,mix_rate)+batch_dot(context[shiftinds],1-mix_rate)
        # dot_points = lambda x,y: torch.einsum('b...,b -> b...',x,y)
        dot_points = lambda x,y: torch.einsum('btd,b -> btd',x,y)
        traj_points = dot_points(traj_points,mix_rate)+dot_points(traj_points[shiftinds],1-mix_rate)
    if config.get('repeat_last', False):
        old_T = context.shape[1]
        context = context[:,-1:].repeat((1, old_T, 1, 1, 1))

    # import pdb; pdb.set_trace()
    # print(m)
    # joined = torch.cat((context,images),axis=1)
    # print(joined.max(),joined.min())
    # flat = rearrange(joined,'b t c h w -> (b h) (t w) c').cpu().numpy()
    # plt.imsave('vis/train_images.jpg',(flat-flat.min())/(flat.max()-flat.min()))
    # import pdb; pdb.set_trace()

    # compute predictions and action LL
    out = m(states, images, context, ret_dist=False,ents=traj['head_label'])
    if 'pred_grasp_point' in out:
        diffs = out['pred_grasp_point']-traj['grasp_point']
        norm = torch.linalg.norm(diffs,axis=1)
        l_readout = norm[traj['start0']].mean()
        if torch.isnan(l_readout):
            l_readout = torch.tensor(0)
    else:
        l_readout = torch.tensor(0)

    if config.get('waypoints',False):
        projection = traj['projection_matrix']
        if config['policy']['vis'].get('context_only',False):
            projection = context_projection
        basegam = config.get('gamma',1e-3) 
        if config.get('anneal_gamma',False):
            progress = trainer._step/trainer.max_batches
            log_dist = np.log10(1e-20/basegam)*progress
            gamma = basegam*(10**log_dist)
        else:
            gamma = basegam
        loss,stats = compute_loss_trajectory(out['waypoints'],
                traj_points,
                projection,
                config.get('num_interp_points',60),
                gamma,
                config['policy'],
                image_waypoints=config.get('image_waypoints',False))
        stats['readout_loss'] = l_readout.item()
        loss += l_readout
        loss_batch = stats.pop('loss_batch')
        for label in set(traj['head_label'].cpu().numpy()):
            label_loss = loss_batch[traj['head_label'] == label].mean().item()
            stats[f'loss/head_{label}'] = label_loss
        if val:
            batch_settings = np.array(traj['setting_name'])
            settings = set(traj['setting_name'])
            for setting in settings:
                mask = batch_settings == setting
                stats[f'split_val_loss/{setting}'] = loss_batch[mask].mean().item()
        return loss,stats
    mu_bc, scale_bc, logit_bc = out['bc_distrib']
    action_distribution = DiscreteMixLogistic(mu_bc[:,:-1], scale_bc[:,:-1], logit_bc[:,:-1])

    l_bc = torch.mean(-action_distribution.log_prob(actions))
    # compute inverse model density
    inv_distribution = DiscreteMixLogistic(*out['inverse_distrib'])
    l_inv = inv_loss_mult * torch.mean(-inv_distribution.log_prob(actions))
        
    # compute goal embedding
    if not goal_loss:
        l_goal, goal_stat = 0, 0
    elif goal_margin < 0:
        l_goal = torch.mean(torch.sum((out['pred_goal'][:,0] - out['img_embed'][:,-1].detach()) ** 2, 1))
        goal_stat = l_goal.item()
    else:
        cos_sims = torch.matmul(out['pred_goal'], out['img_embed'].transpose(1, 2))
        goal_sim, other_sim = cos_sims[:,:,-1], cos_sims[:,0,:-1]
        l_goal = torch.mean(torch.nn.functional.relu(other_sim - goal_sim + goal_margin))
        goal_stat = l_goal.item()

    loss = l_goal + l_inv + l_bc + l_readout
    stats = {'inverse_loss':l_inv.item(), 'bc_loss': l_bc.item(), 'goal_loss': goal_stat,'readout_loss': l_readout.item()}

    if 'point_ll' in out and 'points' in traj:
        pnts = traj['points'].long()
        l_point = torch.mean(-out['point_ll'][range(pnts.shape[0]), pnts[:,-1,0], pnts[:,-1,1]])
        loss = loss + pnt_weight * l_point
        stats['point_loss'] = l_point.item()
        if trainer.is_img_log_step:
            points_img = torch.exp(out['point_ll'].detach())
            maxes = points_img.reshape((points_img.shape[0], -1)).max(dim=1)[0] + 1e-3
            stats['point_img'] = (points_img[:,None] / maxes.reshape((-1, 1, 1, 1))).repeat((1, 3, 1, 1))
            stats['point_img'] = 0.7 * stats['point_img'] + 0.3 * traj['target_images'][:,0]
            pnt_color = torch.from_numpy(np.array([0,1,0])).float().to(stats['point_img'].device).reshape((1, 3))
            for i in range(-5, 5):
                for j in range(-5, 5):
                    h = torch.clamp(pnts[:,-1,0] + i, 0, images.shape[3] - 1)
                    w = torch.clamp(pnts[:,-1,1] + j, 0, images.shape[4] - 1)
                    stats['point_img'][range(pnts.shape[0]),:,h,w] = pnt_color

    stats['bc_loss'] = l_bc.item()
    stats['bc_mse'] = ((actions.cpu().detach() - action_distribution.mean.detach().cpu())**2).mean().item()
    mean_ac = np.clip(action_distribution.mean.detach().cpu().numpy(), -1, 1)
    mean_inv = np.clip(inv_distribution.mean.detach().cpu().numpy(), -1, 1)
    for d in range(actions.shape[2]):
        a_d = actions.cpu().numpy()[:,:,d]
        stats['bc_l1_{}'.format(d)] = np.mean(np.abs(mean_ac[:,:,d] - a_d))
        stats['inv_l1_{}'.format(d)] = np.mean(np.abs(mean_inv[:,:,d] - a_d))
    return loss, stats

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='default')
    parser.add_argument('experiment_file', type=str, help='path to YAML experiment config file',default=None)
    parser.add_argument('--save_path', type=str, default='', help='path to place model save file in during training (overwrites config)')
    parser.add_argument('--save-parent', type=str, default='', help='path to place model save file in during training (overwrites config)')
    parser.add_argument('--device', type=int, default=None, nargs='+', help='target device (uses all if not specified)')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--reg-log', action='store_true')
    parser.add_argument('--nodeterm', action='store_true')
    args = parser.parse_args()

    trainer = Trainer(args,'bc_inv', "Trains Behavior Clone w/ inverse + goal on input data")
    config = trainer.config
    set_seed(trainer.seed,determ=config.get('determ',True) and trainer.determ )
    experiment_log(trainer.save_dir)
    goal_loss, goal_margin = config.get('goal_loss', False), config.get('goal_margin', -1)
    action_model = InverseImitation(**config['policy'])
    if trainer.resume is not None:
        action_model.load_state_dict(torch.load(trainer.resume, map_location=torch.device('cpu')).state_dict())
    inv_loss_mult = config.get('inv_loss_mult', 1.0)
    trainer.train(action_model, forward)
