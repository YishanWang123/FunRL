#!/bin/bash
# run_td3bc_halfcheetah.sh
export CUDA_VISIBLE_DEVICES=0
export WANDB_BASE_URL=https://api.bandw.top

python cleanrl/td3bc.py \
    --env_id "Humanoid-v4" \
    --exp_name "td3_bc_0_human" \
    --total_timesteps 1000000 \
    --buffer_size 1000000 \
    --learning_rate 3e-4 \
    --start_timesteps 25000 \
    --batch_size 256 \
    --gamma 0.99 \
    --tau 0.005 \
    --policy_noise 0.2 \
    --noise_clip 0.5 \
    --policy_freq 2 \
    --eval_episodes 10 \
    --eval_freq 5000 \
    --alpha 0 \
    --seed 1