# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/td3/#td3_continuous_actionpy
"""this is TD3+BC simple implementation based on cleanrl_td3 codebase"""
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
# from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.buffers import ReplayBuffer
from model.MLPPolicy import Actor
from model.MLPCrtitic import Critic

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1  # random seed of the experiment
    torch_deterministic: bool = False  # if toggled, `torch.backends.c
    cuda: bool = True  # if toggled, cuda will be enabled by default
    track: bool = False  # if toggled, this experiment will be tracked with wandb
    wandb_project_name: str = "funrl"  # the wandb's project name
    wandb_entity: str = None  # the entity (team) of wandb's project
    capture_video: bool = False  # whether to capture videos of the agent performance
    save_model: bool = False  # whether to save model
    upload_model: bool = False  # whether to upload model to wandb
    hf_entity: str = None  # the entity (team) of the HuggingFace hub

    """algorithm TD3 + BC Args"""
    env_id: str = "Hopper-v4"  # the id of the environment
    total_timesteps: int = 1000000  # total timesteps of the experiments
    learning_rate: float = 3e-4  # learning rate of the optimizer
    num_envs: int = 1  # the number of parallel game environments
    buffer_size: int = 1000000  # replay buffer size
    gamma: float = 0.99  # discount factor gamma, used in the calculation of the total discounted reward
    batch_size: int = 256  # the batch size of
    tau: float = 0.005  # target smoothing coefficient(τ) for soft update of target parameters
    policy_noise: float = 0.2  # noise added to target policy during critic update
    noise_clip: float = 0.5  # range to clip target policy noise
    policy_freq: int = 2  # frequency of delayed policy updates
    alpha: float = 2.5  # coefficient of behavior cloning loss
    explore_noise: float = 0.1  # std of Gaussian exploration noise
    start_timesteps: int = 25000  # time steps initial random policy is used (optional)
    eval_episodes: int = 10  # number of episodes to evaluate the agent

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(
                env,
                f"videos/{run_name}",
                episode_trigger=lambda x: True,
                disable_logger=True,
            )
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env
    return thunk

if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    if args.capture_video:
        assert args.num_envs == 1, "Cannot capture video when num_envs > 1"

    # writer = SummaryWriter(f"runs/{run_name}")
    # writer.add_text(
    #     "hyperparameters",
    #     "|param|value|\n|-|-|\n%s"
    #     % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    # )

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    if args.torch_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    # action_scale = torch.tensor((envs.single_action_space.high - envs.single_action_space.low) / 2.0, dtype=torch.float32)
    state_dim = int(np.prod(envs.single_observation_space.shape))
    action_dim = int(np.prod(envs.single_action_space.shape))
    #tbd
    actor = Actor(state_dim, action_dim).to(device)
    q_net1 = Critic(state_dim, action_dim).to(device)
    q_net2 = Critic(state_dim, action_dim).to(device)
    actor_target = Actor(state_dim, action_dim).to(device)
    actor_target.load_state_dict(actor.state_dict())

    q_net1_target = Critic(state_dim, action_dim).to(device)
    q_net1_target.load_state_dict(q_net1.state_dict())

    q_net2_target = Critic(state_dim, action_dim).to(device)
    q_net2_target.load_state_dict(q_net2.state_dict())
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.learning_rate)
    q_optimizer = optim.Adam(list(q_net1.parameters()) + list(q_net2.parameters()), lr=args.learning_rate)  #q_net1.parameters(), lr=args.learning_rate)

    envs.single_observation_space.dtype = np.float32
    replay_buffer = ReplayBuffer(args.buffer_size, envs.single_observation_space, envs.single_action_space, device, n_envs=args.num_envs, handle_timeout_termination=False)
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game, reset the environment
    obs, _ = envs.reset(seed=args.seed)
    # obs = torch.Tensor(obs).to(device)
    for global_step in range(args.total_timesteps):
        if global_step < args.start_timesteps:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                actions = actor(torch.Tensor(obs).to(device), deterministic=True)
                #TBD action_scale
                actions += torch.normal(0, args.explore_noise, size=actions.shape).to(device)
                actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)


        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    # writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    # writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    if args.track:
                        episodic_return = info["episode"]["r"]
                        episodic_length = info["episode"]["l"]
                        wandb.log({
                            "charts/episodic_return": episodic_return,
                            "charts/episodic_length": episodic_length,
                            "global_step": global_step
                        })
                    break
        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        replay_buffer.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        if global_step >= args.start_timesteps:
            data = replay_buffer.sample(args.batch_size)
            with torch.no_grad():
                clipped_noise = (torch.randn_like(data.actions, device = device) * args.policy_noise).clamp(
                   -args.noise_clip, args.noise_clip
                ) 
                next_actions = (actor_target(data.next_observations, deterministic=True) + clipped_noise).clamp(
                    envs.single_action_space.low[0], envs.single_action_space.high[0]
                )
                target_q1_values = q_net1_target(data.next_observations, next_actions)
                target_q2_values = q_net2_target(data.next_observations, next_actions)
                target_q_values = torch.min(target_q1_values, target_q2_values)
                target_q_values = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * target_q_values
            
            q1_curr_values = q_net1(data.observations, data.actions).flatten()
            q2_curr_values = q_net2(data.observations, data.actions).flatten()
            q1_loss = F.mse_loss(q1_curr_values, target_q_values)
            q2_loss = F.mse_loss(q2_curr_values, target_q_values)
            q_loss = q1_loss + q2_loss

            q_optimizer.zero_grad()
            q_loss.backward()
            q_optimizer.step()

            ### actor update
            if global_step % args.policy_freq == 0:
                actor_loss = (-q_net1(data.observations, actor(data.observations,deterministic=True))).mean()
                bc_loss = F.mse_loss(actor(data.observations, deterministic=True), data.actions)
                total_loss = actor_loss +args.alpha * bc_loss
                actor_optimizer.zero_grad()
                total_loss.backward()
                actor_optimizer.step()
                if global_step % 100 == 0:
                    if args.track:
                        wandb.log({
                            "losses/q1_loss": q1_loss.item(),
                            "losses/q2_loss": q2_loss.item(),
                            "losses/q_loss": q_loss.item() / 2.0,
                            "losses/actor_loss": actor_loss.item(),
                            "losses/bc_loss": bc_loss.item(),
                            "charts/SPS": int(global_step / (time.time() - start_time)),
                            "global_step": global_step
                        })
                for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(q_net1.parameters(), q_net1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(q_net2.parameters(), q_net2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.funrl_model"
        torch.save(actor.state_dict(), model_path)
        print(f"model save to {model_path}")
        from cleanrl_utils.evals.td3_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name = f"{run_name}_eval",
            Model = (Actor, Critic),
            device = device,
            exploration_noise=args.explore_noise,
        )

    envs.close()






