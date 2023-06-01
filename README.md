# One-shot Visual Imitation via Attributed Waypoints and Demonstration Augmentation
This is the code for [One-shot Visual Imitation via Attributed Waypoints and Demonstration Augmentation](https://arxiv.org/abs/2302.04856). This code was developed using python 3.7.4 with pytorch 1.13 on a CUDA 11.2 machine. 
## Installation
Once you have the appropriate CUDA setup for your hardware, install python packages `pip install -r requirements`.
Install and setup [mujoco](https://github.com/deepmind/mujoco). This code is tested with mujoco version 2.10
This repo relies on code modified code from [Robosuite](https://github.com/SudeepDasari/robosuite), [T-TOSIL](https://github.com/SudeepDasari/one_shot_transformers), [Metaworld](https://github.com/Farama-Foundation/Metaworld) and [MOSAIC](https://github.com/rll-research/mosaic). These components have already been included in this repo.

# Generate Data
Choose a location to store rendered data `export DATA_LOCATION=...`
To generate data for the pick and place task, run the following two commands
```
python scripts/collect_demonstrations.py --env PandaPickPlaceDistractor --num_workers 30 --N 1600 --collect_cam --per_task_group 100 --n_env 800 $DATA_LOCATION/panda
python scripts/collect_demonstrations.py --env SawyerPickPlaceDistractor --num_workers 30 --N 1600 --collect_cam --per_task_group 100 --n_env 800 $DATA_LOCATION/sawyer
```
To generate task driven and high entropy data for data for metaworld
```
python scripts/gen_metaworld_data.py --start 0 --end 100 --output $DATA_LOCATION/metaworld
python scripts/gen_metaworld_data.py --trajectory-synthesis --start 0 --end 100 --output $DATA_LOCATION/metaworld_ts
```

If you experience import errors, you may need to add the current directory to your python path `export PYTHONPATH=$PYTHONPATH:.`

To label grasps on the metaworld dataset
```
python ./scripts/detect_grasps_metaworld.py $DATA_LOCATION/metaworld
```

## Training
Set the data location you chose above and gpus for training. Launch with the specified config file.
```
# pick and place data
EXPERT_DATA=$DATA_LOCATION CUDA_VISIBLE_DEVICES=0,1 python scripts/train_transformer.py experiments/pick_place_simple.yaml

# metworld (including trajecotry synthesis data)
EXPERT_DATA=$DATA_LOCATION CUDA_VISIBLE_DEVICES=0,1 python scripts/train_transformer.py experiments/metaworld_with_ts.yaml
```
Tensorboard logs and model checkpoints are saved into `./outputs`

## Evaluation
To run evaluation on held out tasks for pick place:
```
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py output/pick_place_simple/bc_inv_ckpt-[TIMESTAMP] --eval-only --write-images --rest
```
`[TIMESTAMP]` needs to be replaced with the appropriate value generated for the training run to be evaluated.

for evaluating on metaworld
```
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py output/metaworld_with_ts/bc_inv_ckpt-[TIMESTAMP] --eval-only --write-images --rest
```

This will write reults for each saved checkpoint into tensorboard and into an sqlite3 local database. To run an example script which reads the values out of this database.

```
python scripts/read_data.py output/pick_place_simple/bc_inv_ckpt-[TIMESTAMP]/log.db
```
