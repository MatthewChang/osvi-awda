from hem.util import parse_basic_config
import torch
from torch.utils.data import DataLoader
import argparse
from hem.datasets import get_dataset
from multiprocessing import cpu_count
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import datetime
import torch.nn as nn
import os
import shutil
import copy
import yaml
from hem.models.lr_scheduler import build_scheduler
import torchvision
import time
import math
import torch.utils.data
from pyutil import sorted_file_match

def dict_to_device(dic,device):
    for k,v in dic.items(): 
        if isinstance(v,torch.Tensor):
            dic[k] = v.to(device)
    return dic

class Profiler(object):
    def __init__(self):
        self.elapsed = 0
        self.start_time = 0
    def start(self):
        self.start_time = time.time()
    def stop(self):
        if self.start_time == 0:
            raise Exception(f'not started')
        self.elapsed += time.time() - self.start_time
        self.start_time = 0
    def reset(self):
        self.elapsed = 0
        self.start_time = 0
    def __enter__(self):
        self.start()
    def __exit__(self):
        self.stop()

class Trainer:
    def __init__(self, args, save_name='train', description="Default model trainer", drop_last=False, allow_val_grad=False):
        now = datetime.datetime.now()
        self.seed = args.seed
        self.determ = not args.nodeterm
        if args.resume:
            if os.path.isdir(args.experiment_file):
                fils,nums = sorted_file_match(args.experiment_file,'bc.*?(\d+)')
                if len(fils) > 0:
                    args.experiment_file = os.path.join(args.experiment_file,fils[-1])
                    print(f'Loading last run {args.experiment_file}')
                fils,inds = sorted_file_match(args.experiment_file,r'model_save-(\d+).pt')
                args.experiment_file = os.path.join(args.experiment_file,fils[-1])
            base_folder = os.path.dirname(args.experiment_file)
            self.resume = args.experiment_file
            args.experiment_file = os.path.join(base_folder,'config.yaml')
            if 'slurm' in args.experiment_file:
                os.system("sed -i 's@/projects/bbhm/mc48@${EXPERT_DATA}@' %s"%(args.experiment_file))
            os.system("sed -i 's@/dev/shm/mc48@${EXPERT_DATA}@' %s"%(args.experiment_file))
        else:
            self.resume=None
        if args.experiment_file is None:
            raise Exception(f'experiment_file required')

        self._config = parse_basic_config(args.experiment_file)
        self.reg_log = args.reg_log
        save_config = copy.deepcopy(self._config)
        if args.save_path:
            self._config['save_path'] = args.save_path
        elif args.save_parent:
            name = os.path.splitext(os.path.basename(args.experiment_file))[0]
            self._config['save_path'] = os.path.join(args.save_parent,name)
        else:
            name = os.path.splitext(os.path.basename(args.experiment_file))[0]
            self._config['save_path'] = os.path.join('output',name)

        # initialize device
        def_device = 0 if args.device is None else args.device[0]
        self._device = torch.device("cuda:{}".format(def_device))
        self._device_list = args.device
        self._allow_val_grad = allow_val_grad

        # parse dataset class and create train/val loaders
        # dataset_class = get_dataset(self._config['dataset'].pop('type'))
        # dataset = dataset_class(**self._config['dataset'], mode='train')
        # val_dataset = dataset_class(**self._config['dataset'], mode='val')
        # if 'dataset_aux' in self._config:
            # aux_dataset_config = {**self._config['dataset'],**self._config['dataset_aux']}
            # aux_dataset = dataset_class(**aux_dataset_config, mode='train')
            # dataset = torch.utils.data.ConcatDataset((dataset,aux_dataset))
            # val_aux_dataset = dataset_class(**aux_dataset_config, mode='val')
            # val_dataset = torch.utils.data.ConcatDataset((val_dataset,val_aux_dataset))
        dataset_class = get_dataset(self._config['dataset'].pop('type'))
        dataset = dataset_class(**self._config['dataset'], mode='train')
        val_dataset = dataset_class(**self._config['dataset'], mode='test')
        if 'aux_datasets' in self._config:
            aux_configs = self._config['aux_datasets']
        elif 'dataset_aux' in self._config:
            aux_configs = [self._config['dataset_aux']]
        else:
            aux_configs = []
        if len(aux_configs) > 0:
            aux_configs = [{**self._config['dataset'],**aux} for aux in aux_configs]
            train_datasets = []
            val_datasets = []
            for aux in aux_configs:
                tds= dataset_class(**aux, mode='train')
                train_datasets.append(dataset_class(**aux, mode='train'))
                val_datasets.append(dataset_class(**aux, mode='val'))
            for ds in (dataset,*train_datasets):
                print(len(ds), ds.agent_dir)
            dataset = torch.utils.data.ConcatDataset((dataset,*train_datasets))
            val_dataset = torch.utils.data.ConcatDataset((val_dataset,*val_datasets))

        if args.workers is not None:
            workers = args.workers
        else:
            workers = self._config.get('loader_workers', cpu_count())
        self._train_loader = DataLoader(dataset, batch_size=self._config['batch_size'], shuffle=True, num_workers= workers, drop_last=drop_last)
        self._val_loader = DataLoader(val_dataset, batch_size=self._config['batch_size'], shuffle=True, num_workers=workers, drop_last=True)

        # set of file saving
        # save_dir = os.path.join(self._config.get('save_path', './'), '{}_ckpt-{}-{}_{}-{}-{}'.format(save_name, now.hour, now.minute, now.day, now.month, now.year))
        save_dir = os.path.join(self._config.get('save_path', './'), '{}_ckpt-{}'.format(save_name, int(now.timestamp())))
        if args.resume:
            save_dir = os.path.dirname(args.experiment_file)
            self.optim_load_path = os.path.join(save_dir,'model_save-optim'+os.path.basename(self.resume)[10:])
            print(f"Resuming {self.resume} in {save_dir}")
        save_dir = os.path.expanduser(save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save_dir = save_dir
        with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:
            yaml.dump(save_config, f, default_flow_style=False)
        self._writer = SummaryWriter(log_dir=os.path.join(save_dir, 'log'))
        self._save_fname = os.path.join(save_dir, 'model_save')
        self._step = None

    @property
    def config(self):
        return copy.deepcopy(self._config)

    def train(self, model, train_fn, weights_fn=None, val_fn=None, save_fn=None, optim_weights=None,no_parallel=False):
        # wrap model in DataParallel if needed and transfer to correct device
        time_freq = 100
        start_time = time.time()
        if self.device_count > 1 and not no_parallel:
            model = nn.DataParallel(model, device_ids=self.device_list)
        model = model.to(self._device)
        
        # initializer optimizer and lr scheduler
        optim_weights = optim_weights if optim_weights is not None else model.parameters()
        optimizer, scheduler = self._build_optimizer_and_scheduler(optim_weights)
        if self.resume is not None:
            optimizer.load_state_dict(torch.load(self.optim_load_path))
        # initialize constants:
        epochs = self._config.get('epochs', 1)
        batches = self._config.get('batches', 0)
        if batches > 0:
            epochs = math.ceil(batches/len(self._train_loader))
        self.max_batches = self._config.get('batches', epochs*len(self._train_loader))
        vlm_alpha = self._config.get('vlm_alpha', 0.6)
        log_freq = self._config.get('log_freq', 20)
        self._img_log_freq = img_log_freq = self._config.get('img_log_freq', 500)
        assert img_log_freq % log_freq == 0, "log_freq must divide img_log_freq!"
        save_freq = self._config.get('save_freq', 5000)

        if val_fn is None:
            val_fn = train_fn

        if self.resume is not None:
            self._step = int(os.path.basename(self.resume)[11:-3])
        else:
            self._step = 0
        train_stats = {'loss': 0}
        val_iter = iter(self._val_loader)
        vl_running_mean = None
        loader_profiler = Profiler()
        train_profiler = Profiler()
        for e in range(epochs):
            print('save path: ', self.save_dir)
            loader_profiler.start()
            for inputs in self._train_loader:
                loader_profiler.stop()
                train_profiler.start()
                self._zero_grad(optimizer)
                inputs[0] = dict_to_device(inputs[0],self._device)
                inputs[1] = dict_to_device(inputs[1],self._device)
                loss_i, stats_i = train_fn(self._config,model, self._device, *inputs)
                self._step_optim(loss_i, self._step, optimizer)
                
                # calculate iter stats
                mod_step = self._step % log_freq
                train_stats['loss'] = (self._loss_to_scalar(loss_i) + mod_step * train_stats['loss']) / (mod_step + 1)
                for k, v in stats_i.items():
                    if isinstance(v, torch.Tensor):
                        assert len(v.shape) >= 4, "assumes 4dim BCHW image tensor!"
                        train_stats[k] = v
                    if k not in train_stats:
                        train_stats[k] = 0
                    train_stats[k] = (v + mod_step * train_stats[k]) / (mod_step + 1)
                
                if mod_step == 0:
                    try:
                        val_inputs = next(val_iter)
                    except StopIteration:
                        val_iter = iter(self._val_loader)
                        val_inputs = next(val_iter)
                    val_inputs[0] = dict_to_device(val_inputs[0],self._device)
                    val_inputs[1] = dict_to_device(val_inputs[1],self._device)

                    if self._allow_val_grad:
                        model = model.eval()
                        val_loss, val_stats = val_fn(self._config,model, self._device, *val_inputs,val=True)
                        model = model.train()
                        val_loss = self._loss_to_scalar(val_loss)
                    else:
                        with torch.no_grad():
                            model = model.eval()
                            val_loss, val_stats = val_fn(self._config,model, self._device, *val_inputs,val=True)
                            model = model.train()
                            val_loss = self._loss_to_scalar(val_loss)

                    # update running mean stat
                    if vl_running_mean is None:
                        vl_running_mean = val_loss
                    vl_running_mean = val_loss * vlm_alpha + vl_running_mean * (1 - vlm_alpha)

                    self._writer.add_scalar('loss/val', val_loss, self._step)
                    for stats_dict, mode in zip([train_stats, val_stats], ['train', 'val']):
                        for k, v in stats_dict.items():
                            if isinstance(v, torch.Tensor) and self.step % img_log_freq == 0:
                                if len(v.shape) == 5:
                                    self._writer.add_video('{}/{}'.format(k, mode), v.cpu(), self._step)
                                else:
                                    v_grid = torchvision.utils.make_grid(v.cpu(), padding=5)
                                    self._writer.add_image('{}/{}'.format(k, mode), v_grid, self._step)
                            elif not isinstance(v, torch.Tensor):
                                self._writer.add_scalar('{}/{}'.format(k, mode), v, self._step)
                    
                    # add learning rate parameter to log
                    lrs = np.mean([p['lr'] for p in optimizer.param_groups])
                    self._writer.add_scalar('lr', lrs, self._step)

                    # flush to disk and print
                    self._writer.file_writer.flush()
                    print('epoch {3}/{4}, step {0}: loss={1:.4f} \t val loss={2:.4f}'.format(self._step, train_stats['loss'], vl_running_mean, e, epochs))
                elif not self.reg_log:
                    print('step {0}: loss={1:.4f}'.format(self._step, train_stats['loss']), end='\r')
                self._step += 1

                if self._step % save_freq == 0:
                    if save_fn is not None:
                        save_fn(self._save_fname, self._step)
                    else:
                        save_module = model
                        if weights_fn is not None:
                            save_module = weights_fn()
                        elif isinstance(model, nn.DataParallel):
                            save_module = model.module
                        torch.save(save_module, self._save_fname + '-{}.pt'.format(self._step))
                    if self._config.get('save_optim', False):
                        torch.save(optimizer.state_dict(), self._save_fname + '-optim-{}.pt'.format(self._step))
                train_profiler.stop()
                if self._step % time_freq == 0:
                    elapsed = time.time()-start_time
                    self._writer.add_scalar('profile/batch_per_second',time_freq/elapsed, self._step)
                    self._writer.add_scalar('profile/time_loading',loader_profiler.elapsed/time_freq, self._step)
                    self._writer.add_scalar('profile/time_training',train_profiler.elapsed/time_freq, self._step)
                    print("bps: %0.2f load: %0.3f train: %0.3f" % (time_freq/elapsed,loader_profiler.elapsed/time_freq,train_profiler.elapsed/time_freq),flush=True)
                    loader_profiler.reset()
                    train_profiler.reset()
                    start_time = time.time()
                loader_profiler.start()
            scheduler.step(val_loss=vl_running_mean)

    @property
    def device_count(self):
        if self._device_list is None:
            return torch.cuda.device_count()
        return len(self._device_list)

    @property
    def device_list(self):
        if self._device_list is None:
            return [i for i in range(torch.cuda.device_count())]
        return copy.deepcopy(self._device_list)

    @property
    def device(self):
        return copy.deepcopy(self._device)

    def _build_optimizer_and_scheduler(self, optim_weights):
        # changed to adamW for improved weight decay
        print("WD: ",self._config.get('weight_decay', 0))
        optimizer = torch.optim.AdamW(optim_weights, self._config['lr'], weight_decay=self._config.get('weight_decay', 0))
        return optimizer, build_scheduler(optimizer, self._config.get('lr_schedule', {}))

    def _step_optim(self, loss, step, optimizer):
        loss.backward()
        optimizer.step()

    def _zero_grad(self, optimizer):
        optimizer.zero_grad()

    def _loss_to_scalar(self, loss):
        return loss.item()

    @property
    def step(self):
        if self._step is None:
            raise Exception("Optimization has not begun!")
        return self._step

    @property
    def is_img_log_step(self):
        return self._step % self._img_log_freq == 0
