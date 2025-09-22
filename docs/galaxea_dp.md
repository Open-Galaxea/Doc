# 🤖 Galaxea DP

GalaxeaDP is the open implementation of the diffusion policy algorithm, compatible with `GalaxeaManipSim` and `GalaxeaLeRobot`. It enables end-to-end pipelines for the Galaxea R1 series.

## ✨ Features
- Full simulation pipeline from data to evaluation
- One-click scripts
- LeRobot dataset format
- Open-loop evaluation
- EE and joint control modes

## 🚀 Installation
```
conda create -n opengalaxea python=3.10
conda activate opengalaxea

git clone https://github.com/OpenGalaxea/GalaxeaDP.git
cd GalaxeaDP
pip install -r requirements.txt

cd ..
git clone https://github.com/OpenGalaxea/GalaxeaManipSim.git
cd GalaxeaManipSim
pip install --no-deps -e .

cd ..
git clone https://github.com/OpenGalaxea/GalaxeaLeRobot.git
cd GalaxeaLeRobot
pip install --no-deps -e .
```

## ⚡ Quick Start
```
bash scripts/sim_r1pro_blocks.sh
```

## 🛠 Data Generation & Conversion
- Collect demos in `GalaxeaManipSim`
- Replay with IK to filter and collect images
- Convert to LeRobot format
```
python -m galaxea_sim.scripts.collect_demos --env-name $env --num-demos 100 --datasets_dir data
python -m galaxea_sim.scripts.replay_demos --env-name $env --target_controller_type bimanual_relaxed_ik --num-demos 100 --dataset_dir data
python -m galaxea_sim.scripts.convert_single_galaxea_sim_to_galaxea_lerobot --task $env --tag replayed --robot r1_pro --use_eef --dataset_dir $dataset_dir
```

### Optional: Download Data/Models
```
# Data
gdown --folder https://drive.google.com/drive/folders/1qBY5OuTXrP3r1H8X9duUekyTEfgDFwvb
# Models
gdown --folder https://drive.google.com/drive/folders/1zWgMPXyWryNDz7pHTJzlwPjoszMGd3q1
```

## 🧠 Policy Sample
```python
sample = dict(
    obs=dict(
        left_ee_pose=torch.Tensor(obs_size, 7),
        right_ee_pose=torch.Tensor(obs_size, 7),
        left_arm=torch.Tensor(obs_size, arm_size),
        right_arm=torch.Tensor(obs_size, arm_size),
        left_gripper=torch.Tensor(obs_size, 1),
        right_gripper=torch.Tensor(obs_size, 1),
        episode_start_left_ee_pose=torch.Tensor(1, 7),
        episode_start_right_ee_pose=torch.Tensor(1, 7),
        head_rgb=torch.Tensor(obs_size, 3, H, W),
        left_wrist_rgb=torch.Tensor(obs_size, 3, H, W),
        right_wrist_rgb=torch.Tensor(obs_size, 3, H, W),
    ),
    action=dict(
        left_ee_pose=torch.Tensor(chunk_size, 7),
        right_ee_pose=torch.Tensor(chunk_size, 7),
        left_arm=torch.Tensor(chunk_size, arm_size),
        right_arm=torch.Tensor(chunk_size, arm_size),
        left_gripper=torch.Tensor(chunk_size, 1),
        right_gripper=torch.Tensor(chunk_size, 1),
    ),
    gt_action=action,
)
```

## 🛠 Training
```
export GALAXEA_DP_WORK_DIR=out
bash train.sh trainer.devices=[0,1,2,3] task=sim/R1ProBlocksStackEasy_eef
```

## 📊 Evaluation
```
bash eval_lerobot.sh trainer.devices=[0] task=open_galaxea/<robot>/<task> \
  ckpt_path=out/open_galaxea/<robot>/<task>/<time>/checkpoints/step_20000.ckpt

bash eval_sim.sh trainer.devices=[0] task=sim/<task> \
  ckpt_path=out/sim/<task>/checkpoints/step_20000.ckpt
```

## 📂 Dataset (R1Pro example)
```python
"observation.images.head_rgb": (360, 640, 3)
"observation.images.left_wrist_rgb": (480, 640, 3)
"observation.images.right_wrist_rgb": (480, 640, 3)
"observation.state.left_arm": (7,)
"observation.state.left_arm.velocities": (7,)
"observation.state.right_arm": (7,)
"observation.state.right_arm.velocities": (7,)
"observation.state.left_ee_pose": (7,)
"observation.state.right_ee_pose": (7,)
"observation.state.left_gripper": (1,)
"observation.state.right_gripper": (1,)
"observation.state.chassis": (3,)
"observation.state.chassis.velocities": (6,)
"observation.state.torso": (4,)
"observation.state.torso.velocities": (4,)
"action.left_ee_pose": (7,)
"action.right_ee_pose": (7,)
"action.left_arm": (7,)
"action.right_arm": (7,)
"action.left_gripper": (1,)
"action.right_gripper": (1,)
```

## License & Acknowledgements
- MIT License
- Built on Diffusion Policy, LeRobot, Robotwin

## Citation
```
@inproceedings{GalaxeaDP,
  title={Galaxea Diffusion Policy},
  author={Galaxea Team},
  year={2025}
}
```
