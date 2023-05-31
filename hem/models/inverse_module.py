import torch
import torch.nn as nn
import torch.nn.functional as F
from hem.models import get_model
from hem.models.traj_embed import NonLocalLayer, TemporalPositionalEncoding, TempConvLayer 
from archs.PositionalEncoding import PositionalEncoding
from torchvision import models
from hem.models.discrete_logistic import DiscreteMixLogistic
import numpy as np
from torch.distributions import MultivariateNormal
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from hem.models.full_transformer import FullTransformer
from hem.models.full_connect_ours import FullConnectOurs
from pyutil import to_channel_last,to_channel_first

class _TransformerFeatures(nn.Module):
    def __init__(self, latent_dim, context_T=3, embed_hidden=256, dropout=0.2, n_st_attn=0, use_ss=True, st_goal_attn=False, use_pe=False, attn_heads=1, attn_ff=128, just_conv=False,linear=False,frozen=False,scratch=False,r3m18=False,normalize=True,cnn=False,context_only=False,cnn2=False,small_head=False,hand_cam=False):
        super().__init__()
        model_name = 'r3m' if r3m18 else 'resnet'
        self._resnet18 = get_model(model_name)(output_raw=True, drop_dim=2, use_resnet18=True,frozen=frozen,scratch=scratch)
        if hand_cam:
            self._depth_resnet18 = get_model(model_name)(output_raw=True, drop_dim=2, use_resnet18=True,frozen=frozen,scratch=scratch)       
            self.late_fusion = nn.Sequential(nn.Linear(1024,512),nn.Dropout(dropout),nn.ReLU())
        self.hand_cam = hand_cam
        self.normalize = normalize
        self.cnn = cnn
        self.cnn2 = cnn2
        ProcLayer = TempConvLayer if just_conv else NonLocalLayer
        self._temporal_process = nn.Sequential(ProcLayer(512, 512, attn_ff, dropout=dropout, n_heads=attn_heads), nn.Conv3d(512, 512, (context_T, 1, 1), 1))
        in_dim, self._use_ss = 1024 if use_ss else 512, use_ss
        self._pe = TemporalPositionalEncoding(512, dropout) if use_pe else None
        if linear or small_head:
            # self._to_embed = nn.Sequential(nn.Dropout(dropout),nn.Linear(in_dim, latent_dim))
            self._to_embed = nn.Sequential(nn.Linear(in_dim, latent_dim))
        else:
            self._to_embed = nn.Sequential(nn.Linear(in_dim, embed_hidden), nn.Dropout(dropout), nn.ReLU(), nn.Linear(embed_hidden, latent_dim))
        if cnn:
            self.channel_reduction = nn.Linear(512,64,bias=False)
            channels = [64*11,512,embed_hidden]
            layers = []
            for ch1,ch2 in zip(channels,channels[1:]):
                layers.append(nn.Conv2d(ch1,ch2,(3,3),padding=(1,1),bias=False))
                layers.append(nn.BatchNorm2d(ch2))
                layers.append(nn.ReLU(inplace=True))
            self.convs = nn.Sequential(*layers)
            self._to_embed = nn.Linear(embed_hidden*3,latent_dim)
        elif cnn2:
            channels = [512,64,embed_hidden]
            layers = []
            for ch1,ch2 in zip(channels,channels[1:]):
                layers.append(nn.Conv2d(ch1,ch2,(3,3),padding=(1,1),bias=False))
                layers.append(nn.BatchNorm2d(ch2))
                layers.append(nn.ReLU(inplace=True))
            self.convs = nn.Sequential(*layers)
            self._pe = PositionalEncoding(embed_hidden*3,dropout=0)
            self._to_embed = nn.Linear(embed_hidden*3*11,latent_dim)
        self._st_goal_attn = st_goal_attn
        self._st_attn = nn.Sequential(*[ProcLayer(512, 512, 128, dropout=dropout, causal=True, n_heads=attn_heads) for _ in range(n_st_attn)])
        self.linear = linear
        self.context_only=context_only

    def forward(self, images, context, forward_predict,ret_resnet=False):
        assert len(images.shape) == 5, "expects [B, T, C, H, W] tensor!"
        # if self.context_only:
            # im_in = context
        # else:
            # im_in = torch.cat((context, images), 1) if forward_predict or self._st_goal_attn or self.cnn or self.cnn2 else images
        resnet_features = self._resnet_features(images,context,forward_predict)
        if self.cnn:
            x = rearrange(resnet_features,'b t c h w -> b t h w c')
            x = self.channel_reduction(x)
            x = rearrange(x,'b t h w c-> b (t c) h w')
            feats = self.convs(x)
            spatial_positions,mask = map(lambda x: x.squeeze(1),self._spatial_embed(feats.unsqueeze(1),ret_mask=True))
            shifted_feats = torch.cat((feats[:,1:],feats[:,0:1]),axis=1)
            selected_feats = torch.mean(shifted_feats*mask,(-1,-2))
            joined = torch.cat((spatial_positions,selected_feats),axis=1)
            return self._to_embed(joined).unsqueeze(1)
        if self.cnn2:
            b,t,c,h,w = resnet_features.shape
            x = rearrange(resnet_features,'b t c h w -> (b t) c h w')
            x = self.convs(x)
            feats = rearrange(x,'(b t) c h w -> b t c h w',t=t)
            spatial_positions,mask = self._spatial_embed(feats,ret_mask=True)
            shifted_feats = torch.cat((feats[:,:,1:],feats[:,:,0:1]),axis=2)
            selected_feats = torch.mean(shifted_feats*mask,(-1,-2))
            joined = torch.cat((spatial_positions,selected_feats),axis=-1)
            if self._pe is not None:
                joined = self._pe(joined)
            joined = rearrange(joined,'b t d->b (t d)')
            return self._to_embed(joined).unsqueeze(1)
        if self.linear:
            # causal mean over features
            features = torch.cat([torch.mean(resnet_features[:,:i+1],1,keepdim=True) for i in range(resnet_features.shape[1])],axis=1)
        else:
            features = self._st_attn(resnet_features.transpose(1, 2)).transpose(1, 2)

        if forward_predict:
            features = self._temporal_process(features.transpose(1, 2)).transpose(1, 2)
            features = torch.mean(features, 1, keepdim=True)
        elif self._st_goal_attn and not self.context_only:
            T_ctxt = context.shape[1]
            features = features[:,T_ctxt:]
        embed = self._to_embed(self._spatial_embed(features))
        if self.normalize:
            embed = F.normalize(embed,dim=2)
        if ret_resnet:
            return embed, resnet_features
        else:
            return embed
    
    def _spatial_embed(self, features,ret_mask=False):
        if not self._use_ss:
            return torch.mean(features, (3, 4))

        features = F.softmax(features.reshape((features.shape[0], features.shape[1], features.shape[2], -1)), dim=3).reshape(features.shape)
        # could also get values (depth?) by doing torch.sum(features*softmaxed_features)
        h = torch.sum(torch.linspace(-1, 1, features.shape[3]).view((1, 1, 1, -1)).to(features.device) * torch.sum(features, 4), 3)
        w = torch.sum(torch.linspace(-1, 1, features.shape[4]).view((1, 1, 1, -1)).to(features.device) * torch.sum(features, 3), 3)
        if ret_mask:
            return torch.cat((h, w), 2), features
        else:
            return torch.cat((h, w), 2)

    def _resnet_features(self, images,context,forward_predict):
        if self.hand_cam:
            cim = self._resnet18(context)
            imf = self._resnet18(images[:,:,:3])
            camf = self._depth_resnet18(images[:,:,3:])
            joined = torch.cat((imf,camf),axis=-3)
            im_features = to_channel_first(self.late_fusion(to_channel_last(joined)))
            features = torch.cat((cim,im_features),axis=1).transpose(1,2)
        else:
            if self.context_only:
                im_in = context
            else:
                im_in = torch.cat((context, images), 1) if forward_predict or self._st_goal_attn or self.cnn or self.cnn2 else images
            if self._pe is None or self.cnn2:
                return self._resnet18(im_in)
            features = self._resnet18(im_in).transpose(1, 2)
            # features: torch.Size([30, 512, 11, 8, 10])
        features = self._pe(features).transpose(1, 2)
        return features


class _AblatedFeatures(_TransformerFeatures):
    def __init__(self, latent_dim, model_type='basic', temp_convs=False, lstm=False, context_T=2):
        nn.Module.__init__(self)
        
        # initialize visual network
        assert model_type in ('basic', 'resnet'), "Unsupported model!"
        self._visual_model = get_model('resnet')(output_raw=True, drop_dim=2, use_resnet18=True) if model_type == 'resnet' else get_model('basic')()
        ss_dim = 1024 if model_type == 'resnet' else 64
        self._use_ss, self._latent_dim = True, latent_dim

        # seperate module to process context if needed
        self._temp_convs = temp_convs
        if temp_convs:
            tc_dim = int(ss_dim / 2)
            fc1, a1 = nn.Linear(ss_dim, ss_dim), nn.ReLU(inplace=True)
            fc2, a2 = nn.Linear(ss_dim, ss_dim), nn.ReLU(inplace=True)
            self._fcs = nn.Sequential(fc1, a1, fc2, a2)
            self._tc = nn.Conv1d(ss_dim, tc_dim, context_T, stride=1)
        else:
            tc_dim = 0
        
        # go from input features to latent vector
        self._to_goal = nn.Sequential(nn.Linear((context_T + 1) * (tc_dim + ss_dim), tc_dim + ss_dim), nn.ReLU(inplace=True), nn.Linear(tc_dim + ss_dim, latent_dim))
        self._to_latent = nn.Sequential(nn.Linear(tc_dim + ss_dim, latent_dim), nn.ReLU(inplace=True), nn.Linear(latent_dim, latent_dim))

        # configure lstm network for sequential processing
        self._has_lstm = lstm
        if lstm:
            self._lstm_module = nn.LSTM(latent_dim, latent_dim, 1)
    
    def forward(self, images, context, forward_predict):
        feats = self._visual_model(torch.cat((context, images), 1))
        feats = self._spatial_embed(feats)

        if self._temp_convs:
            ctxt_feats = feats[:,:context.shape[1]]
            ctxt_feats = self._tc(self._fcs(ctxt_feats).transpose(1, 2)).transpose(1,2)
            feats = torch.cat((feats, ctxt_feats.repeat((1, feats.shape[1], 1))), 2)
        
        if forward_predict:
            goal_feats = feats[:,:context.shape[1] + 1].reshape((feats.shape[0], -1))
            return self._to_goal(goal_feats)[:,None]

        latents = self._to_latent(feats)
        latents = self._lstm(latents) if self._has_lstm else latents
        return latents[:,context.shape[1]:]

    def _lstm(self, latents):
        assert self._has_lstm, "needs lstm to forward!"
        self._lstm_module.flatten_parameters()
        return self._lstm_module(latents)[0]


class _DiscreteLogHead(nn.Module):
    def __init__(self, in_dim, out_dim, n_mixtures, const_var=False):
        super().__init__()
        assert n_mixtures >= 1, "must predict at least one mixture!"
        self._n_mixtures, self._dist_size = n_mixtures, torch.Size((out_dim, n_mixtures))
        self._mu = nn.Linear(in_dim, out_dim * n_mixtures)
        if const_var:
            ln_scale = torch.randn(out_dim, dtype=torch.float32) / np.sqrt(out_dim)
            self.register_parameter('_ln_scale', nn.Parameter(ln_scale, requires_grad=True))
        else:
            self._ln_scale = nn.Linear(in_dim, out_dim * n_mixtures)
        self._logit_prob = nn.Linear(in_dim, out_dim * n_mixtures) if n_mixtures > 1 else None
    
    def forward(self, x):
        mu = self._mu(x).reshape((x.shape[:-1] + self._dist_size))
        if isinstance(self._ln_scale, nn.Linear):
            ln_scale = self._ln_scale(x).reshape((x.shape[:-1] + self._dist_size))
        else:
            ln_scale = self._ln_scale if self.training else self._ln_scale.detach()
            ln_scale = ln_scale.reshape((1, 1, -1, 1)).expand_as(mu)
        
        logit_prob = self._logit_prob(x).reshape((x.shape[:-1] + self._dist_size)) if self._n_mixtures > 1 else torch.ones_like(mu)
        return (mu, ln_scale, logit_prob)


class InverseImitation(nn.Module):
    def __init__(self, latent_dim, lstm_config, sdim=9, adim=8, n_mixtures=3, concat_state=True, const_var=False, pred_point=False, vis=dict(), transformer_feat=True, waypoints=None,no_goal=False,linear=False,sub_waypoints=False,grasp=True,ent_head=False,num_ent_head=2):
        super().__init__()
        if sub_waypoints and waypoints is not None:
            waypoints = (waypoints+1)*( waypoints)//2
        self.waypoints = waypoints
        non_linearity = nn.Identity if linear else nn.ReLU
        self.ent_head = ent_head
        self.num_ent_head = num_ent_head
        # initialize visual embeddings
        self._embed = _TransformerFeatures(latent_dim, **vis) if transformer_feat else _AblatedFeatures(latent_dim, **vis)
        if waypoints is not None:
            if ent_head:
                self.waypoint_head = nn.Sequential(nn.Dropout(0.2),nn.ReLU(),nn.Linear(latent_dim,waypoints*4*self.num_ent_head),Rearrange('... (h w d) -> ... h w d',d = 4,h=self.num_ent_head))
            else:
                self.waypoint_head = nn.Sequential(nn.Linear(lstm_config['out_dim'],waypoints*4),Rearrange('... (w d) -> ... w d',d = 4))
        # All of the below model code and weights is only for the action conditioned baseline from T-OSIL
        # inverse modeling
        inv_dim = latent_dim * 2
        self._inv_model = nn.Sequential(nn.Linear(inv_dim, lstm_config['out_dim']), nn.ReLU())

        # additional point loss on goal
        self._2_point = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, 5)) if pred_point else None
        self.no_goal = no_goal

        # action processing
        assert n_mixtures >= 1, "must predict at least one mixture!"
        self._concat_state, self._n_mixtures = concat_state, n_mixtures
        self._is_rnn = lstm_config.get('is_rnn', True)
        ac_size = int(latent_dim + float(concat_state) * sdim)
        # If goal embed is added extend size ac_size
        if not self.no_goal: ac_size += latent_dim
        if self._is_rnn:
            self._action_module = nn.LSTM(ac_size, lstm_config['out_dim'], lstm_config['n_layers'])
        else:
            l1, l2 = [nn.Linear(ac_size, lstm_config['out_dim']), non_linearity()], []
            for _ in range(lstm_config['n_layers'] - 1):
                l2.extend([nn.Linear(lstm_config['out_dim'], lstm_config['out_dim']), nn.ReLU()])
            self._action_module = nn.Sequential(*(l1 + l2))
        act_scale = num_ent_head if ent_head and waypoints is None else 1
        self._action_dist = _DiscreteLogHead(lstm_config['out_dim'], adim*act_scale, n_mixtures, const_var)

    def forward(self, states, images, context, ret_dist=True,ents=None):
        if ents is None:
            ents=torch.zeros((context.shape[0]),device=context.device)
        if self.waypoints is not None:
            images = images[:,:-1,:]
        img_embed = self._embed(images, context, False)
        if self.waypoints is not None:
            out = {}
            if self.ent_head:
                x = self.waypoint_head(img_embed[:,-1])
                inds = repeat(ents,'b -> b 1 w d',w=x.shape[-2],d=x.shape[-1]).long()
                out['waypoints'] = torch.gather(x,1,inds).squeeze(1)
            else:
                out['waypoints'] = rearrange(img_embed[:,-1,:self.waypoints*4],'b (w d) -> b w d',d=4)
            return out
        else:
            print("Running baseline code with no waypoints")
            pred_latent, goal_embed = self._pred_goal(images[:,:1], context)
            states = torch.cat((img_embed, states), 2) if self._concat_state else img_embed

            # run inverse model
            inv_in = torch.cat((img_embed[:,:-1], img_embed[:,1:]), 2)
            mu_inv, scale_inv, logit_inv = self._action_dist(self._inv_model(inv_in))
            if self.ent_head:
                # split actions based on head label
                mu, sc,lo = [rearrange(x,'b t (heads d) mix -> b t heads d mix',heads=self.num_ent_head) for x in (mu_inv,scale_inv,logit_inv)]
                b,t,h,d,m = mu.shape
                inds = repeat(ents,'b -> b t 1 d m',b=b,t=t,d=d,m=m).long()
                mu_inv,scale_inv,logit_inv = [torch.gather(x,2,inds).squeeze(2) for x in (mu,sc,lo)]

            if self.no_goal:
                ac_in = states.transpose(0,1)
            else:
                ac_in = goal_embed.transpose(0, 1).repeat((states.shape[1], 1, 1))
                ac_in = torch.cat((ac_in, states.transpose(0, 1)), 2)
            if self._is_rnn:
                self._action_module.flatten_parameters()
                ac_pred = self._action_module(ac_in)[0].transpose(0, 1)
            else:
                ac_pred = self._action_module(ac_in.transpose(0, 1))
            mu_bc, scale_bc, logit_bc = self._action_dist(ac_pred)
            if self.ent_head:
                # split actions based on head label
                mu_bc_split, scale_split,logit_split = [rearrange(x,'b t (heads d) mix -> b t heads d mix',heads=self.num_ent_head) for x in (mu_bc,scale_bc,logit_bc)]
                b,t,h,d,m = mu_bc_split.shape
                inds = repeat(ents,'b -> b t 1 d m',b=b,t=t,d=d,m=m).long()
                mu_bc,scale_bc,logit_bc = [torch.gather(x,2,inds).squeeze(2) for x in (mu_bc_split,scale_split,logit_split)]
            out = {}

            # package distribution in objects or as tensors
            if ret_dist:
                out['bc_distrib'] = DiscreteMixLogistic(mu_bc, scale_bc, logit_bc)
                out['inverse_distrib'] = DiscreteMixLogistic(mu_inv, scale_inv, logit_inv)
            else:
                out['bc_distrib'] = (mu_bc, scale_bc, logit_bc)
                out['inverse_distrib'] = (mu_inv, scale_inv, logit_inv)

            out['pred_goal'] = pred_latent
            out['img_embed'] = img_embed
            self._pred_point(out, goal_embed, images.shape[3:])
            return out

    # only used in action conditioned baseline
    def _pred_goal(self, img0, context):
        g_embed = self._embed(img0, context, True)
        return g_embed, g_embed

    # only used in action conditioned baseline
    def _pred_point(self, obs, goal_embed, im_shape, min_std=0.03):
        if self._2_point is None:
            return
        
        point_dist = self._2_point(goal_embed[:,0])
        mu = point_dist[:,:2]
        c1, c2, c3 = F.softplus(point_dist[:,2])[:,None], point_dist[:,3][:,None], F.softplus(point_dist[:,4])[:,None]
        scale_tril = torch.cat((c1 + min_std, torch.zeros_like(c2), c2, c3 + min_std), dim=1).reshape((-1, 2, 2))
        mu, scale_tril = [x.unsqueeze(1).unsqueeze(1) for x in (mu, scale_tril)]
        point_dist = MultivariateNormal(mu, scale_tril=scale_tril)

        h = torch.linspace(-1, 1, im_shape[0]).reshape((1, -1, 1, 1)).repeat((1, 1, im_shape[1], 1))
        w = torch.linspace(-1, 1, im_shape[1]).reshape((1, 1, -1, 1)).repeat((1, im_shape[0], 1, 1))
        hw = torch.cat((h, w), 3).repeat((goal_embed.shape[0], 1, 1, 1)).to(goal_embed.device)
        obs['point_ll'] = point_dist.log_prob(hw)
