import glob, math, imageio
import re
from einops.einops import rearrange, repeat
import torch.nn as nn
def read_ims(folder,extension='jpg'):
    ims = []
    num = len(glob.glob(f'{folder}/*.jpg'))
    dig = math.ceil(math.log10(num))
    for i in range(num):
        ims.append(imageio.imread(f'{folder}/%0{dig}d.{extension}' % (i)))
    return ims

import math
import pathlib
import numpy as np
def write_sequence(folder,ims,pad_num=True,extension='jpg'):
    import numpy as np
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
    dig = math.ceil(math.log10(len(ims)))
    for i,im in enumerate(ims):
        if pad_num:
            fname = f'{folder}/%0{dig}d.{extension}' % (i)
        else:
            fname = f'{folder}/{i}.{extension}'
        if isinstance(im,np.ndarray):
            import imageio
            imageio.imwrite(fname,im)
        else:
            im.save(fname, extension)

# interable, comprabarable metric function -> max index, max element, max value
def argmax(li,func = lambda x: x):
    index, max_val,max_el = None,None,None
    for i,el in enumerate(li):
        val = func(el)
        if max_val is None or val > max_val:
            index, max_val,max_el = i, val,el
    return index,max_el,max_val

# interable, comprabarable metric function -> max index, max element, max value
def argmin(li,func = lambda x: x):
    ind,el,val = argmax(li,lambda x: -func(x))
    return ind,el,-val

# reproducability snippet
import sys
import os
# Only works correctly if running in the git root directory
def experiment_log(log_folder):
    pathlib.Path(log_folder).mkdir(parents=True, exist_ok=True)
    # unstaged changes (files in source control)
    os.system(f'git diff > {log_folder}/code_diff.txt')
    # staged changes
    os.system(f'git diff --cached >> {log_folder}/code_diff.txt')
    # unstaged changes (files not source control)
    # Only works correctly if running in the git root directory
    os.system(f'git ls-files --others --exclude-standard | while read -r i; do git diff -- /dev/null "$i"; done >> {log_folder}/code_diff.txt')
    # check for binary files
    ret = os.system(f"grep -e '^Binary files .* and .* differ$' {log_folder}/code_diff.txt")
    if ret == 0:
        raise Exception(f'Binary files found in git diff')
    # log current commit
    os.system(f'echo "Commit: " > {log_folder}/info.txt')
    os.system(f'git rev-parse --verify HEAD >> {log_folder}/info.txt')
    with open(f"{log_folder}/info.txt", "a") as fil: fil.write(' '.join(sys.argv)) 
    with open(f"{log_folder}/args.txt", "a") as fil: fil.write(' '.join(sys.argv[1:])) 
    class Tee(object):
        def __init__(self, name, mode):
            self.file = open(name, mode)
            self.stdout = sys.stdout
            sys.stdout = self
        def __del__(self):
            sys.stdout = self.stdout
            self.file.close()
        def write(self, data):
            self.file.write(data)
            self.stdout.write(data)
        def flush(self):
            self.file.flush()
    TEEVAR = Tee(f'{log_folder}/log.txt','w')

from functools import wraps
import inspect
def initializer(func):
    """
    Automatically assigns the parameters.

    import util.initializer
    >>> class process:
    ...     @util.initializer
    ...     def __init__(self, cmd, reachable=False, user='root'):
    ...         pass
    >>> p = process('halt', True)
    >>> p.cmd, p.reachable, p.user
    ('halt', True, 'root')
    """
    names, varargs, keywords, defaults = inspect.getargspec(func)
    if defaults is None: defaults = []

    @wraps(func)
    def wrapper(self, *args, **kargs):
        for name, arg in list(zip(names[1:], args)) + list(kargs.items()):
            setattr(self, name, arg)

        for name, default in zip(reversed(names), reversed(defaults)):
            if not hasattr(self, name):
                setattr(self, name, default)
        func(self, *args, **kargs)
    return wrapper

import torch.nn as nn
import torch
class GaussianNetwork(nn.Module):
    def __init__(self):
        super(GaussianNetwork, self).__init__()
    def forward(self,x):
        dim = x.shape[-1]
        assert dim % 2 == 0
        means = x[...,:dim//2]
        stds = torch.exp(x[...,dim//2:])+1e-8
        return torch.distributions.normal.Normal(means,stds)

from  torch.distributions.transformed_distribution import TransformedDistribution
from  torch.distributions.transforms import SigmoidTransform, AffineTransform
from  torch.distributions import Uniform
class LogisticDistribution(torch.distributions.Distribution):
    def __init__(self,loc,scale):
        super().__init__(validate_args=False)
        self.loc = loc
        self.scale = scale
        base_distribution = Uniform(0, 1)
        transforms = [SigmoidTransform().inv, AffineTransform(loc=loc, scale=scale)]
        self.td = TransformedDistribution(base_distribution, transforms)
        self.td.base_dist.low = self.td.base_dist.low.to(loc.device)
        self.td.base_dist.high = self.td.base_dist.high.to(scale.device)
    @property
    def mean(self):
        return self.loc

    def log_prob(self, value):
        return self.td.log_prob(value)

class LogisticNetwork(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        dim = x.shape[-1]
        assert dim % 2 == 0
        means = x[...,:dim//2]
        scales = torch.exp(x[...,dim//2:])+1e-8
        return LogisticDistribution(means,scales)

class Logistic(TransformedDistribution):
    def __init__(self,loc,scale):
        super().__init__(Uniform(torch.zeros_like(loc), torch.ones_like(loc)), [SigmoidTransform().inv, AffineTransform(loc, scale)])
        self.loc = loc
        self.scale = scale

    @property 
    def mean(self):
        return self.loc

def loopGenerator(loader):
    i = iter(loader)
    while True:
        try:
            yield next(i)
        except StopIteration:
            i = iter(loader)

import argparse
def build_name(args,parser,skip=[]):
    # most common usage
    # name = build_name(args,parser,skip=['prefix'])
    # name = args.prefix + f"_{name}" if args.prefix else name
    string = ""
    e = parser._optionals._actions[-1]
    for e in parser._optionals._actions:
        if e.dest in ['help']+skip:
            continue
        if getattr(args,e.dest) != e.default:
            name = e.dest
            name = name.replace('_','')
            string += f"_{name}" if len(string) > 0 else name
            if isinstance(e, argparse._StoreAction):
                val = getattr(args,e.dest)
                if isinstance(val,list):
                    val='-'.join(val)
                string += str(val)
            else:
                assert isinstance(e, argparse._StoreTrueAction)
    return string

# gives files sorted based on first capture group of pattern
def sorted_file_match(directory,pat):
    files = os.listdir(directory)
    matches = [re.match(pat,e) for e in files]
    pairs = [(fil,int(match[1])) for fil,match in zip(files,matches) if match]
    if len(pairs) == 0:
        return [[],[]]
    return list(zip(*sorted(pairs,key=lambda x: x[1])))

import random
def set_seed(seed,determ=True):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(determ)
    torch.backends.cudnn.deterministic = determ
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

def rollout_general(env,act,render_ims = False,is_vec_env=False,progress=None,render_function=None):
    states = []
    ims = []
    actions = []
    rewards = []
    infos = []
    state = env.reset()
    states.append(state)
    if render_function:
        render = render_function
    else:
        render = lambda: np.stack(env.get_images()) if is_vec_env else env.render('rgb')
    if render_ims:
        ims.append(render())
    done = None
    total_rew = 0
    info = None
    def end_rollouts():
        if done is None:
            return False
        if is_vec_env:
            return done.any()
        else:
            return done
    while not end_rollouts():
        with torch.no_grad():
            action,policy_done = act(state)
        # augment ims with attention vis
        # step env
        state,reward,done,info = env.step(action)
        done = done or policy_done
        im3 = env.render()
        if render_ims:
            ims.append(render())
        # env.reset()
        # env.render()
        # env.sim.forward()
        # im1 = env.render()
        # # plt.imsave('vis/test.png',im1)
        # env.step([0,0,0,0])
        # im2 = env.render()
        # plt.imsave('vis/test2.png',np.concatenate([im1,im2],axis=1))
        states += [state]
        actions += [action]
        rewards += [reward]
        infos += [info]
        total_rew += reward
    if render_ims:
        ims = np.array(ims)
    return np.array(states),ims,np.array(actions),np.array(rewards),infos

from gym import ObservationWrapper
from gym.wrappers import FrameStack
from gym.spaces import Box
class LambdaWrapper(ObservationWrapper):
    r"""Observation wrapper that runs a function. The observation_space property won't reflect scale changes."""

    def __init__(self, env,lam,space=None):
        super().__init__(env)
        self.lam = lam
        if space is None:
            self.observation_space = Box(lam(env.observation_space.low),lam(env.observation_space.high))
        else:
            self.observation_space=space


    def observation(self, state):
        return self.lam(state)

def FlatFrameStack(env,num):
    return LambdaWrapper(FrameStack(env,num_stack=num),np.ravel)

from collections import deque
import gym.spaces as spaces
class DictFrameStack(ObservationWrapper):
    r"""Observation wrapper that stacks the observations in a rolling manner.
    For example, if the number of stacks is 4, then the returned observation contains
    the most recent 4 observations. For environment 'Pendulum-v1', the original observation
    is an array with shape [3], so if we stack 4 observations, the processed observation
    has shape [4, 3]."""

    def __init__(self, env, num_stack):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)
        newspace = {}
        for name,space in env.observation_space.spaces.items():
            low = np.repeat(space.low[np.newaxis, ...], num_stack, axis=0)
            high = np.repeat( space.high[np.newaxis, ...], num_stack, axis=0)
            newspace[name]=Box( low=low, high=high, dtype=space.dtype)
        self.observation_space = spaces.Dict(newspace)

    def observation(self):
        assert len(self.frames) == self.num_stack, (len(self.frames), self.num_stack)
        out = {}
        for k in self.observation_space.spaces.keys():
            out[k] = np.stack([ e[k] for e in self.frames])
        return out

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.frames.append(observation)
        return self.observation(), reward, done, info

    def reset(self, **kwargs):
        if kwargs.get("return_info", False):
            obs, info = self.env.reset(**kwargs)
        else:
            obs = self.env.reset(**kwargs)
            info = None  # Unused
        [self.frames.append(obs) for _ in range(self.num_stack)]

        if kwargs.get("return_info", False):
            return self.observation(), info
        else:
            return self.observation()

# use np.array_split for splitting into chunks
from torch.utils import data as tdata
class LambdaLoader(tdata.Dataset):
    def __init__(self,length,func):
        self.length=length
        self.func=func

    def __len__(self):
        'Denotes the total number of samples'
        return self.length

    def __getitem__(self, index):
        return self.func(index)

def mujoco_depth_to_meters(sim, depth):
    extent = sim.model.stat.extent
    near = sim.model.vis.map.znear * extent
    far = sim.model.vis.map.zfar * extent
    image = near / (1 - depth * (1 - near / far))
    return image

class LambdaLayer(nn.Module):
    def __init__(self,func):
        super(LambdaLayer,self).__init__()
        self.func = func
    def forward(self,x):
        return self.func(x)

import scipy.interpolate
def interpolate_lines(data,num_points,ranges=None):
    if ranges is None:
        ranges = data[:,:,0].min(),data[:,:,0].max()
    points = np.linspace(ranges[0],ranges[1],endpoint=True,num=500)
    new_lines = []
    for line in data:
        interp = scipy.interpolate.interp1d(line[:,0],line[:,1],fill_value=(line[0,1],line[-1,1]),bounds_error=False)
        new_lines.append(np.stack((points,interp(points)),axis=1))
    return np.stack(new_lines)

def to_channel_first(images):
    return rearrange(images,'... r col c -> ... c r col')

def to_channel_last(images):
    return rearrange(images,'... c r col -> ... r col c')

def tcf(images): return to_channel_first(images)
def tcl(images): return to_channel_last(images)

