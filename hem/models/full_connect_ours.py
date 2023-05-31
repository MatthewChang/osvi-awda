from archs.PositionalEncoding import PositionalEncoding
from archs.ResnetFeatures import ResnetFeatures
from einops.layers.torch import Rearrange
import torch.nn as nn
import torch
from hem.models import get_model
from einops import rearrange, reduce, repeat
from pyutil import LambdaLayer
def mul(x):
    return x*255

class FullConnectOurs(nn.Module):
    def __init__(
        self,
        action_dim,
        state_dim,
        no_state=False,
        context_state_dim=None,
        state_context=False,
        no_context=False,
        waypoints=10,
        dropout=0,
        r3m18=False,
        pe=True,
        context_only=True,
    ):
        frame_stack = 1 if not context_only else 0
        self.context_only = context_only
        self.state_dim = state_dim
        intermediate_size = 512
        self.state_dim = state_dim
        self.state_context = state_context
        self.no_context = no_context
        self.context_state_dim = context_state_dim or state_dim
        self.action_dim = action_dim
        self.num_waypoints = waypoints
        self.non_linearity = torch.relu
        self.no_state=no_state
        super().__init__()
        extra = frame_stack
        self.pe = pe
        if r3m18:
            self.r3m = get_model('r3m')(drop_dim=1, use_resnet18=True,frozen=True)
            layers = [self.r3m]
            # else:
                # self.r3m = load_r3m("resnet18")
                # # self.r3m = get_model('r3m')(drop_dim=1, use_resnet18=True,frozen=True)
                # # freeze r3m model
                # for param in self.r3m.parameters():
                    # param.requires_grad = False
                # # def do_r3m(x):
                    # # x = rearrange(x*255,'b t c w h -> (b t) c w h')
                    # # x = self.r3m(x)
                    # # x = rearrange(x,'(b t) d -> b t d',t=11)
                    # # return x
                # # layers = [LambdaLayer(do_r3m)]
                # layers = [LambdaLayer(mul),Rearrange('b t c w h -> (b t) c w h'),self.r3m,Rearrange('(b t) d -> b t d',t=11)]
            # outsize = 512 if config.args.r3m18 else 2048
            outsize = 512
            if intermediate_size != outsize: layers.append(nn.Linear(outsize,intermediate_size))
        else:
            layers = [ResnetFeatures(normalize=False,frozen=True)]
            if intermediate_size != 512: layers.append(nn.Linear(512,intermediate_size)) #type: ignore
        if self.pe:
            layers.append(PositionalEncoding(intermediate_size, dropout=dropout))
        else:
            layers.append(nn.Dropout(p=dropout))
        self.image_features = nn.Sequential(*layers)
        context_size = 0 if self.no_context else 10
        self.transformer = nn.Sequential(
                Rearrange('b t d -> b (t d)'),
                nn.Linear((context_size+extra)*intermediate_size,self.num_waypoints*intermediate_size),
                # nn.Linear((10+frame_stack)*39,10*intermediate_size),
                Rearrange('b (t d) -> b t d', t = self.num_waypoints),
                )
        hidden_size = intermediate_size
        inpsize = hidden_size if no_state else hidden_size + self.state_dim*frame_stack
        dist = nn.Linear(inpsize, action_dim)
        self.head = dist

    def forward(self, state, context,ims=None):
        if self.context_only:
            inps = context
        else:
            inps = torch.cat([context,ims],axis=1) #type: ignore
        image_feats = self.image_features(inps)
        res = self.transformer(self.non_linearity(image_feats))
        x = res
        if self.no_state:
            joined = x
        else:
            state_extended = repeat(state,'b d -> b t d', t = x.shape[1])
            joined = torch.cat((x, state_extended), dim=-1)
        return self.head(self.non_linearity(joined))

