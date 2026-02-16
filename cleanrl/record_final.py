"""
Video recording with trained model - using FastLIBEROEnv for proper state
"""
import os
import sys
sys.path.insert(0, '/teamspace/studios/this_studio')

import numpy as np
import torch
import torch.nn as nn
import cv2
import wandb

# Initialize wandb to log video
wandb.init(project="libero-ppo-fast", name="video_eval_final")

# Add paths
sys.path.insert(0, '/teamspace/studios/this_studio/cleanrl')
from fast_libero_env import FastLIBEROEnv


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=19, action_dim=7, hidden_dims=[256, 256]):
        super().__init__()
        
        # Policy network
        policy_layers = []
        in_dim = obs_dim
        for h_dim in hidden_dims:
            policy_layers.extend([nn.Linear(in_dim, h_dim), nn.Tanh()])
            in_dim = h_dim
        self.policy = nn.Sequential(*policy_layers, nn.Linear(in_dim, action_dim))
        
        # Value network
        value_layers = []
        in_dim = obs_dim
        for h_dim in hidden_dims:
            value_layers.extend([nn.Linear(in_dim, h_dim), nn.Tanh()])
            in_dim = h_dim
        self.value = nn.Sequential(*value_layers, nn.Linear(in_dim, 1))
        
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
    def get_eval_action(self, x):
        return self.policy(x)


# Create env using FastLIBEROEnv (same as training)
env = FastLIBEROEnv(
    benchmark_name='libero_spatial',
    task_id=0,
    max_episode_steps=300,
    num_sim_steps=1
)

# Load model
model = ActorCritic(obs_dim=19, action_dim=7)
model.load_state_dict(torch.load('/teamspace/studios/this_studio/cleanrl/runs/model_final.pt', map_location='cpu'))
model.eval()

print("Model loaded!")

# Create env with rendering for video
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import DenseRewardEnv

benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict['libero_spatial']()
task = task_suite.get_task(0)
task_bddl_file = os.path.join(get_libero_path('bddl_files'), task.problem_folder, task.bddl_file)

render_env = DenseRewardEnv(
    bddl_file_name=task_bddl_file,
    use_camera_obs=True,
    has_offscreen_renderer=True,
    camera_heights=480,
    camera_widths=640
)

# Video path
video_file = "/teamspace/studios/this_studio/cleanrl/runs/eval_video_final.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_file, fourcc, 30, (640, 480))

print(f"Saving video to {video_file}")

# Get target object names from render env
inner = render_env.env
target_objects = []
for name in inner.objects_dict.keys():
    if len(target_objects) >= 2:
        break
    target_objects.append(name)

ctrlrange = render_env.sim.model.actuator_ctrlrange[:7]

for episode in range(1):
    render_env.reset()
    obs = env.reset()[0]  # FastLIBEROEnv
    episode_reward = 0
    
    for step in range(300):
        # Get action from model using FastLIBEROEnv state
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).to('cpu')
            action = model.get_eval_action(obs_tensor).numpy()
        
        # Scale action
        action_scaled = np.clip(action, -1, 1)
        action_muj = (action_scaled + 1) / 2 * (ctrlrange[:, 1] - ctrlrange[:, 0]) + ctrlrange[:, 0]
        
        # Step render env for observation
        obs_render, reward, done, info = render_env.step(action_muj)
        episode_reward += reward
        
        # Get image
        if 'agentview_image' in obs_render:
            frame = obs_render['agentview_image']
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)
        
        # Step fast env to get next state
        obs, _, _, _, _ = env.step(action)
        
        if done:
            break
    
    print(f"Episode {episode+1}: reward = {episode_reward:.2f}")

out.release()
render_env.close()
env.close()

print(f"Video saved to {video_file}")

# Log to wandb
wandb.log({"eval_video": wandb.Video(video_file, fps=30), "eval_reward": episode_reward})
wandb.finish()
print("Done!")
