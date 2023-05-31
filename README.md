# One-shot Visual Imitation via Attributed Waypoints and Demonstration Augmentation
This is the preliminary code release for [One-shot Visual Imitation via Attributed Waypoints and Demonstration Augmentation](https://arxiv.org/abs/2302.04856). This code was developed using python 3.7.4 with pytorch 1.13 on a CUDA 11.2 machine. 
## Installation
Once you have the appropriate CUDA setup install python packages `pip install -r requirements`.
Install and setup mujoco. This code is tested with mujoco version 2.10
This repo relies on code modified code from [Robosuite](https://github.com/SudeepDasari/robosuite), [T-TOSIL](https://github.com/SudeepDasari/one_shot_transformers), [Metaworld](https://github.com/Farama-Foundation/Metaworld) and [MOSAIC](https://github.com/rll-research/mosaic). These components have already been included in this repo.

# Generate Data
Choose a location to store rendered data `export DATA_LOCATION=...`
To generate data for the pick and place task, run the following two commands
```
python scripts/collect_demonstrations.py --env PandaPickPlaceDistractor --num_workers 30 --N 1600 --collect_cam --per_task_group 100 --n_env 800 $DATA_LOCATION/panda
python scripts/collect_demonstrations.py --env SawyerPickPlaceDistractor --num_workers 30 --N 1600 --collect_cam --per_task_group 100 --n_env 800 $DATA_LOCATION/sawyer
```

## Training
Set the data location you chose above and gpus for training. Launch with the specified config file.
```
EXPERT_DATA=$DATA_LOCATION CUDA_VISIBLE_DEVICES=0,1 python scripts/train_transformer.py experiments/pick_place_simple.yaml
```
Tensorboard logs and model checkpoints are saved into `./outputs`
## Evaluation
Coming soon
