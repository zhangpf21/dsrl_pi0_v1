#!/bin/bash
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH="$(pwd):$(pwd)/LIBERO:${PYTHONPATH}"

proj_name=DSRL_pi0_Libero_adapter

device_id=0

export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl  
export MUJOCO_EGL_DEVICE_ID=$device_id

export OPENPI_DATA_HOME=./openpi
export EXP=./logs/$proj_name; 
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export NUMBA_DISABLE_JIT=1
export MPLCONFIGDIR=/tmp/matplotlib

python3 -m pip install mujoco==3.3.1 "numpy<2"

python3 -m examples.launch_train_sim \
--algorithm pixel_sac \
--env libero \
--prefix dsrl_pi0_libero_adapter \
--wandb_project ${proj_name} \
--batch_size 256 \
--discount 0.999 \
--seed 0 \
--max_steps 500000  \
--eval_interval 10000 \
--log_interval 500 \
--eval_episodes 10 \
--multi_grad_step 5 \
--start_online_updates 2000 \
--resize_image 64 \
--action_magnitude 1.0 \
--query_freq 5 \
--hidden_dims 128 \
--use_adapter_conditioning 1 \
--noise_dim 32 \
--control_dim 16 \
--adapter_feature_dim 1024 \
--adapter_gate_dim 1 \
--adapter_hidden_dim 128 \
--actor_lr 3e-5 \
--adapter_l2_coef 1e-3 \
--gate_l1_coef 1e-3
