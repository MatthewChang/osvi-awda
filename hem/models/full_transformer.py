from archs.PositionalEncoding import PositionalEncoding
from archs.ResnetFeatures import ResnetFeatures
from einops.layers.torch import Rearrange
import torch.nn as nn
import torch
from einops import rearrange, reduce, repeat
class FullTransformer(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        context_state_dim=None,
        state_context=False,
        no_context=False,
        full_trans=False,
        trajectory=False,
        rgbd=False,
        waypoints=None,
        latent_size=512,
        dropout=0.2,
        no_state=False,
        frame_stack = 1
    ):
        self.no_state = no_state
        intermediate_size = latent_size
        self.state_context = state_context
        self.no_context = no_context
        self.full_trans = full_trans
        self.trajectory = trajectory
        self.context_state_dim = context_state_dim or state_dim
        self.action_dim = action_dim
        if waypoints is None:
            raise Exception(f'no wayopitns')
        self.num_waypoints = waypoints

        self.non_linearity = torch.relu
        super().__init__()
        layers = [ResnetFeatures(normalize=False,depth=rgbd,frozen=False)]
        if intermediate_size != 512: layers.append(nn.Linear(512,intermediate_size)) #type: ignore
        self.image_features = nn.Sequential(*layers, PositionalEncoding(intermediate_size, dropout=dropout))
        self.decoding_tokens=nn.Embedding(self.num_waypoints,intermediate_size)
        self.transformer = nn.Transformer(d_model=intermediate_size, nhead=8, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=intermediate_size, dropout=0.1, activation='relu')

        if self.no_context:
            hidden_size = 0
        else:
            hidden_size = intermediate_size
        inpsize = int(hidden_size if no_state else hidden_size + self.state_dim*frame_stack)
        dist = nn.Linear(inpsize, action_dim)
        self.head = dist

    def forward(self, state, context,ims=None):
        if self.no_context:
            return self.head(state)
        # inps = context if self.config.args.transform_target or self.config.args.noims else torch.cat([context,ims],axis=1) #type: ignore
        inps = torch.cat([context,ims],axis=1) #type: ignore
        image_feats = self.image_features(inps)
        tokens = self.decoding_tokens(torch.arange(self.num_waypoints).to(self.decoding_tokens.weight.device))
        tokens = repeat(tokens,'t d -> t b d', b = image_feats.shape[0])
        batch_second = rearrange(image_feats,'b t d -> t b d')
        res = self.transformer(self.non_linearity(batch_second),tokens)
        res = rearrange(res,'t b d -> b t d')
        if self.trajectory:
            x = res
        else:
            x = res[...,-1,:]
        if self.no_state:
            joined = x
        elif self.trajectory:
            state_extended = repeat(state,'b d -> b t d', t = x.shape[1])
            joined = torch.cat((x, state_extended), dim=-1)
        else:
            joined = torch.cat((x, state), dim=-1)
        return self.head(self.non_linearity(joined))

