#!/bin/bash
proj_name=DSRL_pi0_Aloha
device_id=0

export DISPLAY=:0
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=$device_id

export OPENPI_DATA_HOME=./openpi
export EXP=./logs/$proj_name; 
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false


pip install mujoco==2.3.7

python3 examples/launch_train_sim.py \
--algorithm pixel_sac \
--env aloha_cube \
--prefix dsrl_pi0_aloha \
--wandb_project ${proj_name} \
--batch_size 256 \
--discount 0.999 \
--seed 0 \
--max_steps 3000000  \
--eval_interval 10000 \
--log_interval 500 \
--eval_episodes 10 \
--multi_grad_step 20 \
--start_online_updates 1000 \
--resize_image 64 \
--action_magnitude 2.0 \
--query_freq 50 \
--hidden_dims 128 \
--use_adapter_conditioning 1 \
--noise_dim 32 \
--control_dim 16 \
--adapter_feature_dim 1024 \
--adapter_gate_dim 1 \
--adapter_hidden_dim 128 \
--adapter_l2_coef 1e-4 \
--gate_l1_coef 1e-4 \
--target_entropy 0.0
