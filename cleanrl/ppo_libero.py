# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
"""
PPO for LIBERO benchmark.
Trains a policy to solve LIBERO-Spatial tasks using PPO with state-based or image-based input.
"""
import os
import sys

# Add LIBERO to path
LIBERO_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "LIBERO")
if LIBERO_PATH not in sys.path:
    sys.path.insert(0, LIBERO_PATH)

import random
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
import gym
from torch import distributions as dist
from torch.utils.tensorboard import SummaryWriter

# LIBERO imports
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv, DenseRewardEnv


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # LIBERO specific arguments
    benchmark_name: str = "libero_spatial"
    """LIBERO benchmark: libero_spatial, libero_object, libero_goal, libero_90, libero_10"""
    task_id: int = 0
    """task ID within the benchmark suite"""
    use_image: bool = False
    """if True, use image observations. If False (default), use state features (much faster)"""
    dense_reward: bool = True
    """if True, use dense reward function. If False (default), use sparse reward"""
    camera_height: int = 128
    """height of the rendered image (only used if use_image=True)"""
    camera_width: int = 128
    """width of the rendered image (only used if use_image=True)"""
    
    # Algorithm specific arguments
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments (currently only 1 supported for LIBERO)"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    max_episode_steps: int = 300
    """maximum steps per episode"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(args, idx=0, seed=0):
    """Create a LIBERO environment."""
    def thunk():
        # Get benchmark and task
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[args.benchmark_name]()
        task = task_suite.get_task(args.task_id)
        
        # Get BDDL file path
        from libero.libero import get_libero_path
        task_bddl_file = os.path.join(
            get_libero_path("bddl_files"), 
            task.problem_folder, 
            task.bddl_file
        )
        
        # Create environment - use DenseRewardEnv for dense rewards
        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": args.camera_height if args.use_image else 128,
            "camera_widths": args.camera_width if args.use_image else 128
        }
        if args.dense_reward:
            env = DenseRewardEnv(**env_args)
        else:
            env = OffScreenRenderEnv(**env_args)
        
        # Set seed
        env.seed(seed + idx)
        
        # Get task initial states for consistent resets
        init_states = task_suite.get_task_init_states(args.task_id)
        
        return env, init_states[0] if len(init_states) > 0 else None
    
    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class StateEncoder(nn.Module):
    """MLP encoder for state-based observations (robot state + object state)."""
    def __init__(self, state_dim, feature_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            layer_init(nn.Linear(state_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, feature_dim)),
            nn.Tanh(),
        )
    
    def forward(self, x):
        return self.encoder(x)


class CNNEncoder(nn.Module):
    """CNN encoder for image-based observations."""
    def __init__(self, in_channels=3, feature_dim=256, image_size=128):
        super().__init__()
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(in_channels, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Calculate feature size based on image size
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, image_size, image_size)
            n_features = self.conv(dummy).shape[1]
        
        self.fc = nn.Sequential(
            layer_init(nn.Linear(n_features, feature_dim)),
            nn.Tanh(),
        )
    
    def forward(self, x):
        # x: (B, C, H, W)
        x = x / 255.0 if x.max() > 1.0 else x  # Normalize if not already
        x = x * 2.0 - 1.0  # Normalize to [-1, 1]
        return self.fc(self.conv(x))


class Agent(nn.Module):
    def __init__(self, observation_shape, action_dim, feature_dim=256, use_image=False, image_size=128, state_dim=None):
        super().__init__()
        self.use_image = use_image
        
        if use_image:
            self.encoder = CNNEncoder(3, feature_dim, image_size)
        else:
            self.encoder = StateEncoder(state_dim, feature_dim)
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(feature_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(feature_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, x):
        features = self.encoder(x)
        return self.critic(features)

    def get_action_and_value(self, x, action=None):
        features = self.encoder(x)
        action_mean = self.actor_mean(features)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = dist.Normal(action_mean, action_std)
        
        if action is None:
            action = probs.sample()
        
        # Clip action to valid range [-1, 1]
        action = torch.clamp(action, -1.0, 1.0)
        
        log_prob = probs.log_prob(action).sum(dim=-1)
        entropy = probs.entropy().sum(dim=-1)
        
        return action, log_prob, entropy, self.critic(features)


class LIBEROWrapper:
    """Wrapper to make LIBERO compatible with the PPO training loop."""
    def __init__(self, env, init_state=None, max_steps=300, use_image=False, image_shape=(3, 128, 128)):
        self.env = env
        self.init_state = init_state
        self.max_steps = max_steps
        self.use_image = use_image
        self.current_step = 0
        
        # Get action space from the environment's spec or sim
        if hasattr(env, 'action_space'):
            self.action_space = env.action_space
        elif hasattr(env, 'sim') and hasattr(env.sim, 'actuators'):
            import gym
            actuator_ctrlrange = env.sim.model.actuator_ctrlrange
            low = actuator_ctrlrange[:, 0]
            high = actuator_ctrlrange[:, 1]
            self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        else:
            import gym
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        
        # Get state dimension for state-based encoding
        # Sample observation to get state dimension
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        
        if use_image:
            # Use image observation space
            self.observation_space = gym.spaces.Box(
                low=0, high=255, 
                shape=image_shape, 
                dtype=np.uint8
            )
            self.state_dim = None
        else:
            # Use state features - extract proprio state and object state
            if isinstance(obs, dict):
                # Get the dimension of state features
                proprio_keys = ['robot0_proprio-state']
                object_keys = ['object-state']
                
                proprio_dim = 0
                if 'robot0_proprio-state' in obs:
                    proprio_dim = len(obs['robot0_proprio-state'])
                elif 'robot0_joint_pos' in obs:
                    # Alternative: construct from individual keys
                    proprio_dim = len(obs['robot0_joint_pos']) + len(obs.get('robot0_joint_vel', [0]))
                    proprio_dim += len(obs.get('robot0_eef_pos', [0])) * 2  # pos + quat
                    proprio_dim += len(obs.get('robot0_gripper_qpos', [0])) * 2  # gripper pos + vel
                
                object_dim = 0
                if 'object-state' in obs and obs['object-state'] is not None:
                    object_dim = len(obs['object-state'])
                else:
                    # Count all object-related keys
                    for k, v in obs.items():
                        if 'pos' in k.lower() or 'quat' in k.lower() or 'state' in k.lower():
                            if isinstance(v, (np.ndarray, list)) and not isinstance(v, str):
                                object_dim += len(v) if hasattr(v, '__len__') else 1
                
                self.state_dim = proprio_dim + object_dim
                print(f"State dimension: proprio={proprio_dim}, object={object_dim}, total={self.state_dim}")
            else:
                self.state_dim = 100  # Fallback default
            
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(self.state_dim,), 
                dtype=np.float32
            )
        
        self._reset()
    
    def _get_state_features(self, obs):
        """Extract state features from observation dict."""
        if isinstance(obs, dict):
            # Use proprioceptive state and object state
            features = []
            
            # Robot proprioceptive state
            if 'robot0_proprio-state' in obs:
                features.append(np.array(obs['robot0_proprio-state']))
            else:
                # Construct from individual components
                proprio = []
                if 'robot0_joint_pos' in obs:
                    proprio.extend(obs['robot0_joint_pos'])
                if 'robot0_joint_pos_cos' in obs:
                    proprio.extend(obs['robot0_joint_pos_cos'])
                if 'robot0_joint_pos_sin' in obs:
                    proprio.extend(obs['robot0_joint_pos_sin'])
                if 'robot0_joint_vel' in obs:
                    proprio.extend(obs['robot0_joint_vel'])
                if 'robot0_eef_pos' in obs:
                    proprio.extend(obs['robot0_eef_pos'])
                if 'robot0_eef_quat' in obs:
                    proprio.extend(obs['robot0_eef_quat'])
                if 'robot0_gripper_qpos' in obs:
                    proprio.extend(obs['robot0_gripper_qpos'])
                if 'robot0_gripper_qvel' in obs:
                    proprio.extend(obs['robot0_gripper_qvel'])
                features.append(np.array(proprio))
            
            # Object state
            if 'object-state' in obs and obs['object-state'] is not None:
                features.append(np.array(obs['object-state']))
            
            return np.concatenate(features) if features else np.zeros(self.state_dim)
        else:
            return np.zeros(self.state_dim)
    
    def _reset(self):
        result = self.env.reset()
        if isinstance(result, tuple):
            obs = result[0]
        else:
            obs = result
        if self.init_state is not None:
            self.env.set_init_state(self.init_state)
            result = self.env.reset()
            obs = result[0] if isinstance(result, tuple) else result
        self.current_step = 0
        return obs
    
    def reset(self, seed=None):
        return self._reset()
    
    def step(self, action):
        # Clip action to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)
        obs, reward, done, info = self.env.step(action)
        self.current_step += 1
        
        # Truncate if max steps reached
        if self.current_step >= self.max_steps:
            done = True
            info['TimeLimit.truncated'] = True
        
        return obs, reward, done, False, info
    
    def close(self):
        self.env.close()
    
    @property
    def single_observation_space(self):
        return self.observation_space
    
    @property
    def single_action_space(self):
        return self.action_space


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    
    # Create run name
    input_type = "image" if args.use_image else "state"
    run_name = f"{args.benchmark_name}_task{args.task_id}_{input_type}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
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
    
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")
    print(f"Using {'image' if args.use_image else 'state'} observations")

    # Get task info for logging
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.benchmark_name]()
    task = task_suite.get_task(args.task_id)
    task_name = task.name
    task_description = task.language
    print(f"Task: {task_name}")
    print(f"Description: {task_description}")

    # env setup
    env, init_state = make_env(args, idx=0, seed=args.seed)()
    image_shape = (3, args.camera_height, args.camera_width) if args.use_image else None
    env = LIBEROWrapper(env, init_state, max_steps=args.max_episode_steps, use_image=args.use_image, image_shape=image_shape)
    
    # Get action dimension
    action_dim = env.single_action_space.shape[0]
    print(f"Action dimension: {action_dim}")
    
    # Create agent
    if args.use_image:
        agent = Agent(
            observation_shape=env.single_observation_space.shape,
            action_dim=action_dim,
            feature_dim=256,
            use_image=True,
            image_size=args.camera_height
        ).to(device)
    else:
        agent = Agent(
            observation_shape=env.single_observation_space.shape,
            action_dim=action_dim,
            feature_dim=256,
            use_image=False,
            state_dim=env.state_dim
        ).to(device)
    
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Storage
    obs_shape = env.single_observation_space.shape
    obs = torch.zeros((args.num_steps,) + obs_shape).to(device)
    actions = torch.zeros((args.num_steps,) + env.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps,)).to(device)
    rewards = torch.zeros((args.num_steps,)).to(device)
    dones = torch.zeros((args.num_steps,)).to(device)
    values = torch.zeros((args.num_steps,)).to(device)

    # Start
    global_step = 0
    start_time = time.time()
    next_obs = env.reset(seed=args.seed)
    if isinstance(next_obs, tuple):
        next_obs = next_obs[0]
    
    if args.use_image:
        # Handle image observations
        if isinstance(next_obs, dict):
            if 'agentview_image' in next_obs:
                next_obs = next_obs['agentview_image']
            elif 'image' in next_obs:
                next_obs = next_obs['image']
            else:
                for k, v in next_obs.items():
                    if 'image' in k.lower():
                        next_obs = v
                        break
        next_obs = torch.Tensor(next_obs).permute(2, 0, 1).unsqueeze(0).to(device)
    else:
        # Handle state observations
        next_obs = torch.Tensor(env._get_state_features(next_obs)).unsqueeze(0).to(device)

    next_done = torch.zeros(1).to(device)

    # Training loop
    for iteration in range(1, args.num_iterations + 1):
        # Annealing
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1
            obs[step] = next_obs.squeeze(0)
            dones[step] = next_done

            # Action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            
            actions[step] = action
            logprobs[step] = logprob

            # Execute
            next_obs_np, reward, terminations, truncations, infos = env.step(action.cpu().numpy().squeeze(0))
            next_done = torch.tensor(float(terminations or truncations)).to(device).view(-1)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            
            # Convert obs to tensor
            if args.use_image:
                if isinstance(next_obs_np, dict):
                    if 'agentview_image' in next_obs_np:
                        next_obs_np = next_obs_np['agentview_image']
                    elif 'image' in next_obs_np:
                        next_obs_np = next_obs_np['image']
                    else:
                        for k, v in next_obs_np.items():
                            if 'image' in k.lower():
                                next_obs_np = v
                                break
                next_obs = torch.Tensor(next_obs_np).permute(2, 0, 1).unsqueeze(0).to(device)
            else:
                next_obs = torch.Tensor(env._get_state_features(next_obs_np)).unsqueeze(0).to(device)

            # Log episode info
            if terminations or truncations:
                print(f"global_step={global_step}, episodic_return={reward}")
                writer.add_scalar("charts/episodic_return", reward, global_step)
                writer.add_scalar("charts/episodic_length", step + 1, global_step)
                
                # Reset environment
                next_obs_np = env.reset()
                if isinstance(next_obs_np, tuple):
                    next_obs_np = next_obs_np[0]
                if args.use_image:
                    if isinstance(next_obs_np, dict):
                        if 'agentview_image' in next_obs_np:
                            next_obs_np = next_obs_np['agentview_image']
                        elif 'image' in next_obs_np:
                            next_obs_np = next_obs_np['image']
                        else:
                            for k, v in next_obs_np.items():
                                if 'image' in k.lower():
                                    next_obs_np = v
                                    break
                    next_obs = torch.Tensor(next_obs_np).permute(2, 0, 1).unsqueeze(0).to(device)
                else:
                    next_obs = torch.Tensor(env._get_state_features(next_obs_np)).unsqueeze(0).to(device)
                next_done = torch.zeros(1).to(device)

        # GAE
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # Flatten batch
        b_obs = obs.reshape((-1,) + env.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + env.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # PPO update
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # Logging
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        
        sps = int(global_step / (time.time() - start_time))
        print(f"iter={iteration}/{args.num_iterations}, SPS: {sps}")
        writer.add_scalar("charts/SPS", sps, global_step)

    # Save model
    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    env.close()
    writer.close()
