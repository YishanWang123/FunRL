#!/bin/bash
export WANDB_BASE_URL=https://api.bandw.top

python -m cleanrl_utils.benchmark \
    --env-ids PongNoFrameskip-v4 BeamRiderNoFrameskip-v4 BreakoutNoFrameskip-v4 \
    --command "python cleanrl/dqn_atari.py --track --capture_video" \
    --num-seeds 1 \
    --workers 1
