# 🤖 Galaxea Manipulation Simulator

Provides simulation benchmarks, expert demo pipelines, and baseline policies for the Galaxea R1 series. Use it to collect demos, convert to `LeRobot` datasets, and train/evaluate Diffusion Policies.

## ✨ Features
- 30+ environments across 17 tasks
- One-command setup and asset registration
- Joint-space and relaxed-ik (EEF) controllers
- LeRobot-compatible datasets
- Baseline DP training and evaluation with videos/metrics

## 🚀 Installation (Standalone)
Prereq: Linux + CUDA GPU
```
conda create -n galaxea-sim python=3.10 -y
conda activate galaxea-sim
pip install -e .

# Install LeRobot
cd ..
git clone https://github.com/huggingface/lerobot.git
cd lerobot
git checkout a5e0aae13a3efd0080ac7ab6b461980d644014ab
pip install -e .
export PYTHONPATH="your_lerobot_codebase_path:$PYTHONPATH"
```

If installing together with Galaxea-DP, see the Galaxea DP installation.

Download assets:
```
gdown https://drive.google.com/file/d/1ZvtCv1H4FLrse_ElUWzsVDt8xRK4CyaC/
unzip robotwin_models.zip
mv robotwin_models galaxea_sim/assets/
```

Heads-up: If you hit `datasets` cache conflicts, clear `~/.cache/huggingface/datasets`.

## 🎮 Collect Demos
Supported robots, example tasks, and controllers:
- R1: `R1DualBottlesPickEasy` — joint_position / relaxed_ik
- R1 Pro: `R1ProBlocksStackEasy` — joint_position / relaxed_ik
- R1 Lite: `R1LiteBlocksStackEasy` — joint_position

### 1) Generate raw demos (mplib)
```
python -m galaxea_sim.scripts.collect_demos --env-name R1DualBottlesPickEasy --num-demos 100
python -m galaxea_sim.scripts.collect_demos --env-name R1ProBlocksStackEasy --num-demos 100
python -m galaxea_sim.scripts.collect_demos --env-name R1LiteBlocksStackEasy --num-demos 100
```
Default `--obs_mode` is `state`. Data saved under `datasets/<env-name>/<date-time>`.

### 2) Replay demos with controllers
```
python -m galaxea_sim.scripts.replay_demos --env-name R1DualBottlesPickEasy --target_controller_type bimanual_joint_position --num-demos 100
python -m galaxea_sim.scripts.replay_demos --env-name R1ProBlocksStackEasy --target_controller_type bimanual_relaxed_ik --num-demos 100
python -m galaxea_sim.scripts.replay_demos --env-name R1LiteBlocksStackEasy --target_controller_type bimanual_joint_position --num-demos 100
```
Replay filters infeasible IK solutions, stores images/depths under `datasets/<env-name>/final`.

## 🛠 Train Policies
### Convert to LeRobot dataset
For Galaxea DP use `convert_single_galaxea_sim_to_galaxea_lerobot` (add `--use_eef` for relaxed_ik). For LeRobot DP use `convert_single_galaxea_sim_to_lerobot`.
```
# LeRobot DP examples
python -m galaxea_sim.scripts.convert_single_galaxea_sim_to_lerobot --task R1DualBottlesPickEasy --tag final --robot r1
python -m galaxea_sim.scripts.convert_single_galaxea_sim_to_lerobot --task R1ProBlocksStackEasy --tag final --robot r1_pro
python -m galaxea_sim.scripts.convert_single_galaxea_sim_to_lerobot --task R1LiteBlocksStackEasy --tag final --robot r1_lite

# EEF examples
python -m galaxea_sim.scripts.convert_single_galaxea_sim_to_lerobot --robot r1_pro --task R1ProDualBottlesPickEasy --tag final --use_eef
python -m galaxea_sim.scripts.convert_single_galaxea_sim_to_lerobot --robot r1_pro --task R1ProBlocksStackEasy --tag final --use_eef
```
Optional `--use_video` stores images as video. Ensure ffmpeg is installed; change vcodec to `libx264` if needed in `.../site-packages/lerobot/common/datasets/video_utils.py`.

### Data structure after converting
#### Galaxea DP
```
# Images
observation.images.head_rgb: (224, 224, 3)
observation.images.left_wrist_rgb: (224, 224, 3)
observation.images.right_wrist_rgb: (224, 224, 3)
# Depth
observation.depth.head_depth: (224, 224)
# States/actions (EEF controller example; arm_dof=6 for R1/R1Lite, 7 for R1Pro)
observation.state.left_arm_joints: (arm_dof,)
observation.state.left_gripper: (1,)
observation.state.right_arm_joints: (arm_dof,)
observation.state.right_gripper: (1,)
action.left_arm_joints: (arm_dof,)
action.left_gripper: (1,)
action.right_arm_joints: (arm_dof,)
action.right_gripper: (1,)
```
#### LeRobot DP
```
# Images
observation.images.rgb_head: (224, 224, 3)
observation.images.rgb_left_hand: (224, 224, 3)
observation.images.rgb_right_hand: (224, 224, 3)
# States/actions (EEF controller)
observation.state: (2*arm_dof + 2,)
action: (2*arm_dof + 2,)
# States/actions (joints controller)
observation.state: (16,)
action: (16,)
```

### Train and Evaluate
```
python -m galaxea_sim.scripts.train_lerobot_dp_policy --task R1DualBottlesPickEasy
python -m galaxea_sim.scripts.train_lerobot_dp_policy --task R1ProBlocksStackEasy
python -m galaxea_sim.scripts.train_lerobot_dp_policy --task R1LiteBlocksStackEasy

python -m galaxea_sim.scripts.eval_lerobot_dp_policy --task R1DualBottlesPickEasy --pretrained-policy-path outputs/train/R1DualBottlesPickEasy/diffusion/.../checkpoint --target_controller_type bimanual_joint_position
python -m galaxea_sim.scripts.eval_lerobot_dp_policy --task R1ProBlocksStackEasy --pretrained-policy-path outputs/train/R1ProBlocksStackEasy/diffusion/.../checkpoint --target_controller_type bimanual_joint_position
python -m galaxea_sim.scripts.eval_lerobot_dp_policy --task R1LiteBlocksStackEasy --pretrained-policy-path outputs/train/R1LiteBlocksStackEasy/diffusion/.../checkpoint --target_controller_type bimanual_joint_position
```

## 📈 Success Rate (100 rollouts; joints controller)
- R1 Dual Bottles Pick Easy: OpenDP 98%, LeRobot 98%
- R1 Pro Blocks Stack Easy: OpenDP 68%, LeRobot 64%
- R1 Lite Blocks Stack Easy: OpenDP 51%, LeRobot 42%

## License & Acknowledgements
- MIT License
- Built on Diffusion Policy, LeRobot, Robotwin

## Citation
```
@inproceedings{GalaxeaManipSim,
  title={Galaxea Manipulation Simulator},
  author={Galaxea Team},
  year={2025}
}
```
