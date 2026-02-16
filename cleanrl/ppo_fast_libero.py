# docs: https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
"""
PPO for LIBERO using FastLIBEROEnv (~2000 SPS)
"""
import os
import sys

# Add LIBERO path
possible_paths = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "LIBERO"),
    "/teamspace/studios/this_studio/LIBERO",
    os.path.expanduser("~/LIBERO"),
]
for LIBERO_PATH in possible_paths:
    if os.path.exists(LIBERO_PATH):
        if LIBERO_PATH not in sys.path:
            sys.path.insert(0, LIBERO_PATH)
        break

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
import gymnasium as gym
from torch import distributions as dist
from dataclasses import dataclass

# Import fast environment
from fast_libero_env import FastLIBEROEnv


@dataclass
class Args:
    exp_name: str = "ppo_fast_libero"
    seed: int = 0
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "libero-ppo-fast"
    wandb_entity: str = None
    
    # Environment
    benchmark_name: str = "libero_spatial"
    task_id: int = 0
    num_sim_steps: int = 1
    max_episode_steps: int = 300
    
    # Training
    total_timesteps: int = 50000
    learning_rate: float = 3e-4
    num_steps: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 8
    """the number of mini-batches - increased from 4 to 8 for better sample efficiency"""
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    """value function coefficient"""
    exploration_std: float = 0.3
    """exploration noise (standard deviation) - lower = less erratic, more stable"""
    max_grad_norm: float = 0.5
    target_kl: float = None


def make_env(args):
    def thunk():
        env = FastLIBEROEnv(
            benchmark_name=args.benchmark_name,
            task_id=args.task_id,
            max_episode_steps=args.max_episode_steps,
            num_sim_steps=args.num_sim_steps
        )
        return env
    return thunk


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, args=None):
        hidden_dims = [256, 256]
        
        # Policy network
        super().__init__()
        
        # Policy network
        policy_layers = []
        in_dim = obs_dim
        for h_dim in hidden_dims:
            policy_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.Tanh()
            ])
            in_dim = h_dim
        self.policy = nn.Sequential(*policy_layers, nn.Linear(in_dim, action_dim))
        
        # Value network
        value_layers = []
        in_dim = obs_dim
        for h_dim in hidden_dims:
            value_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.Tanh()
            ])
            in_dim = h_dim
        self.value = nn.Sequential(*value_layers, nn.Linear(in_dim, 1))
        
        # Log std for exploration - use config value (lower = less erratic)
        if args and hasattr(args, 'exploration_std'):
            init_log_std = np.log(args.exploration_std)
        else:
            init_log_std = 0.0  # exp(0) = 1.0 std
        self.log_std = nn.Parameter(torch.full((action_dim,), init_log_std))
        
    def get_value(self, x):
        return self.value(x)
    
    def get_action(self, x, deterministic=False):
        mean = self.policy(x)
        std = torch.exp(self.log_std)
        
        if deterministic:
            return mean, None
        
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob
    
    def get_eval_action(self, x):
        return self.policy(x)


def train(args):
    # Setup
    if args.track:
        import wandb
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, name=args.exp_name)
        
        # Log all config arguments
        wandb.config.update(vars(args))
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.backends.cudnn.deterministic = args.torch_deterministic
    
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Environment
    print("Creating environment...")
    env = make_env(args)()
    print("Environment created!")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"Obs dim: {obs_dim}, Action dim: {action_dim}")
    
    # Model - pass args for exploration_std
    model = ActorCritic(obs_dim, action_dim, args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Storage
    obs = np.zeros((args.num_steps, obs_dim), dtype=np.float32)
    actions = np.zeros((args.num_steps, action_dim), dtype=np.float32)
    rewards = np.zeros(args.num_steps, dtype=np.float32)
    dones = np.zeros(args.num_steps, dtype=np.float32)
    values = np.zeros(args.num_steps, dtype=np.float32)
    log_probs = np.zeros(args.num_steps, dtype=np.float32)
    
    # Training loop
    global_step = 0
    episode_count = 0
    episode_return = 0
    
    # Track reward components
    episode_reward_gripper_bowl = 0
    episode_reward_bowl_plate = 0
    episode_reward_height = 0
    
    # Initial observation
    next_obs, _ = env.reset()
    next_obs = next_obs.astype(np.float32)
    
    while global_step < args.total_timesteps:
        if global_step % 1000 == 0:
            print(f"Step {global_step}/{args.total_timesteps}, Episodes: {episode_count}", flush=True)
        
        # Collect rollout
        for step in range(args.num_steps):
            obs[step] = next_obs
            
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(next_obs).to(device)
                action, log_prob = model.get_action(obs_tensor)
                value = model.get_value(obs_tensor)
            
            action_np = action.cpu().numpy()
            log_prob_np = log_prob.cpu().numpy()
            value_np = value.cpu().numpy().item()
            
            actions[step] = action_np
            log_probs[step] = log_prob_np
            values[step] = value_np
            
            # Step environment
            next_obs, reward, done, truncated, info = env.step(action_np)
            next_obs = next_obs.astype(np.float32)
            
            rewards[step] = reward
            dones[step] = done or truncated
            episode_return += reward
            
            # Track reward components
            if isinstance(info, dict):
                episode_reward_gripper_bowl += info.get('reward_gripper_bowl', 0)
                episode_reward_bowl_plate += info.get('reward_bowl_plate', 0)
                episode_reward_height += info.get('reward_height', 0)
            
            global_step += 1
            
            if done or truncated:
                next_obs, _ = env.reset()
                next_obs = next_obs.astype(np.float32)
                
                if args.track:
                    wandb.log({
                        "episode_return": episode_return, 
                        "reward_gripper_bowl": episode_reward_gripper_bowl,
                        "reward_bowl_plate": episode_reward_bowl_plate,
                        "reward_height": episode_reward_height,
                        "global_step": global_step
                    })
                
                episode_return = 0
                episode_reward_gripper_bowl = 0
                episode_reward_bowl_plate = 0
                episode_reward_height = 0
                episode_count += 1
        
        # Compute returns and advantages
        with torch.no_grad():
            last_value = model.get_value(torch.FloatTensor(next_obs).to(device)).cpu().numpy().item()
        
        returns = np.zeros(args.num_steps, dtype=np.float32)
        advantages = np.zeros(args.num_steps, dtype=np.float32)
        
        gae = 0
        for step in reversed(range(args.num_steps)):
            if step == args.num_steps - 1:
                next_non_terminal = 1.0 - dones[step]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - dones[step]
                next_value = values[step + 1]
            
            delta = rewards[step] + args.gamma * next_value * next_non_terminal - values[step]
            gae = delta + args.gamma * args.gae_lambda * next_non_terminal * gae
            advantages[step] = gae
            returns[step] = gae + values[step]
        
        # Flatten
        b_obs = obs.reshape(-1, obs_dim)
        b_actions = actions.reshape(-1, action_dim)
        b_log_probs = log_probs.reshape(-1)
        b_returns = returns.reshape(-1)
        b_advantages = advantages.reshape(-1)
        
        # Normalize advantages
        if args.norm_adv:
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        
        # PPO update
        b_advantages = b_advantages.astype(np.float32)
        
        inds = np.arange(args.num_steps)
        for epoch in range(args.update_epochs):
            np.random.shuffle(inds)
            
            for start in range(0, args.num_steps, args.num_steps // args.num_minibatches):
                end = start + args.num_steps // args.num_minibatches
                mb_inds = inds[start:end]
                
                mb_obs = torch.FloatTensor(b_obs[mb_inds]).to(device)
                mb_actions = torch.FloatTensor(b_actions[mb_inds]).to(device)
                mb_advantages = torch.FloatTensor(b_advantages[mb_inds]).to(device)
                mb_returns = torch.FloatTensor(b_returns[mb_inds]).to(device)
                mb_old_log_probs = torch.FloatTensor(b_log_probs[mb_inds]).to(device)
                
                # Get new log probs and values
                new_actions, new_log_probs = model.get_action(mb_obs)
                new_values = model.get_value(mb_obs).squeeze()
                
                # Policy loss
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.functional.mse_loss(new_values, mb_returns)
                
                # Entropy bonus
                entropy_loss = -torch.distributions.Normal(
                    model.policy(mb_obs), torch.exp(model.log_std)
                ).entropy().mean()
                
                # Total loss
                loss = policy_loss + args.vf_coef * value_loss + args.ent_coef * entropy_loss
                
                # Optimize
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
        
        # Logging
        if args.track:
            wandb.log({
                "loss/policy": policy_loss.item(),
                "loss/value": value_loss.item(),
                "loss/entropy": entropy_loss.item(),
                "global_step": global_step
            })
        
        if global_step % 1000 == 0:
            print(f"Step {global_step}/{args.total_timesteps}, Episodes: {episode_count}")
        
        # Save model periodically
        if global_step % 50000 == 0 and global_step > 0:
            os.makedirs("runs", exist_ok=True)
            torch.save(model.state_dict(), f"runs/model_{global_step}.pt")
            print(f"Saved model at step {global_step}")
    
    # Save final model
    os.makedirs("runs", exist_ok=True)
    torch.save(model.state_dict(), "runs/model_final.pt")
    print("Training complete!")
    
    # Record and log video
    if args.track:
        import cv2
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import DenseRewardEnv
        
        print("Recording evaluation video...")
        try:
            # Create env for recording
            benchmark_dict = benchmark.get_benchmark_dict()
            task_suite = benchmark_dict[args.benchmark_name]()
            task = task_suite.get_task(args.task_id)
            task_bddl_file = os.path.join(get_libero_path('bddl_files'), task.problem_folder, task.bddl_file)
            
            record_env = DenseRewardEnv(
                bddl_file_name=task_bddl_file,
                use_camera_obs=True,
                has_offscreen_renderer=True,
                camera_heights=480,
                camera_widths=640
            )
            
            obs = record_env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            
            ctrlrange = record_env.sim.model.actuator_ctrlrange[:7]
            
            # Video path
            video_file = "runs/eval_video.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_file, fourcc, 30, (640, 480))
            
            episode_reward = 0
            for step in range(300):
                # Get state
                proprio = obs.get('robot0_proprio-state', np.zeros(14)) if isinstance(obs, dict) else np.zeros(14)
                obj_state = obs.get('object-state', np.zeros(12)) if isinstance(obs, dict) else np.zeros(12)
                goal = np.array([0.0, 0.2, 0.0])
                obs_state = np.concatenate([proprio, obj_state[:6], goal]).astype(np.float32)[:26]
                if len(obs_state) < 26:
                    obs_state = np.pad(obs_state, (0, 26 - len(obs_state)))
                
                # Get action from model
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs_state).to(device)
                    action = model.get_eval_action(obs_tensor).cpu().numpy()
                
                action_scaled = np.clip(action, -1, 1)
                action_muj = (action_scaled + 1) / 2 * (ctrlrange[:, 1] - ctrlrange[:, 0]) + ctrlrange[:, 0]
                
                obs, reward, done, info = record_env.step(action_muj)
                if isinstance(obs, tuple):
                    obs = obs[0]
                episode_reward += reward
                
                if isinstance(obs, dict) and 'agentview_image' in obs:
                    frame = obs['agentview_image']
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame)
                
                if done:
                    break
            
            out.release()
            record_env.close()
            
            # Log video
            wandb.log({"eval_video": wandb.Video(video_file, fps=30), "eval_reward": episode_reward})
            print(f"Video logged! Eval reward: {episode_reward}")
        except Exception as e:
            print(f"Video recording failed: {e}")
    
    env.close()
    
    if args.track:
        wandb.finish()


if __name__ == "__main__":
    args = tyro.cli(Args)
    train(args)
