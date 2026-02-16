"""
Video recording with trained model - logs to wandb
"""
import os
import sys
sys.path.insert(0, '/teamspace/studios/this_studio/LIBERO')

import numpy as np
import torch
import torch.nn as nn
import cv2
import wandb

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import DenseRewardEnv

# Initialize wandb
wandb.init(project="libero-ppo-fast", name="video_eval_500k")


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=26, action_dim=7, hidden_dims=[256, 256]):
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


# Create env with camera
benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict['libero_spatial']()
task = task_suite.get_task(0)
task_bddl_file = os.path.join(get_libero_path('bddl_files'), task.problem_folder, task.bddl_file)

env = DenseRewardEnv(
    bddl_file_name=task_bddl_file, 
    use_camera_obs=True,
    has_offscreen_renderer=True,
    camera_heights=480,
    camera_widths=640
)

# Get proper initial obs
obs = env.reset()
if isinstance(obs, tuple):
    obs = obs[0]

# Load model
model = ActorCritic(obs_dim=19, action_dim=7)
model.load_state_dict(torch.load('/teamspace/studios/this_studio/cleanrl/runs/model_final.pt', map_location='cpu'))
model.eval()

print("Model loaded!")

# Video path
video_path = "/teamspace/studios/this_studio/cleanrl/runs/libero_spatial_task0__ppo_fast_libero__1__eval/videos/"
os.makedirs(video_path, exist_ok=True)
video_file = os.path.join(video_path, "eval.mp4")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_file, fourcc, 30, (640, 480))

print(f"Saving video to {video_file}")

ctrlrange = env.sim.model.actuator_ctrlrange[:7]

for episode in range(1):
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    episode_reward = 0
    
    for step in range(300):
        # Get state: robot joint positions + object positions relative to gripper
        sim = env.sim
        gripper_pos = sim.data.body_xpos[env.robot.eef_site_id]
        
        # Joint positions (7)
        qpos = sim.data.qpos[:7]
        
        # Object positions relative to gripper (12 for 2 objects)
        rel_obj_pos = []
        for name in env.target_objects[:2]:
            try:
                obj_pos = sim.data.body_xpos[sim.model.body_name2id(name)]
                rel_pos = obj_pos - gripper_pos
                rel_obj_pos.append(rel_pos)
            except:
                rel_obj_pos.append(np.zeros(3))
        
        # Combine: 7 + 12 = 19 dims
        obs_state = np.concatenate([qpos] + rel_obj_pos).astype(np.float32)
        
        # Get action
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs_state)
            action = model.get_eval_action(obs_tensor).numpy()
        
        # Scale action
        action_scaled = np.clip(action, -1, 1)
        action_muj = (action_scaled + 1) / 2 * (ctrlrange[:, 1] - ctrlrange[:, 0]) + ctrlrange[:, 0]
        
        # Step
        obs, reward, done, info = env.step(action_muj)
        if isinstance(obs, tuple):
            obs = obs[0]
        episode_reward += reward
        
        # Get image
        if isinstance(obs, dict) and 'agentview_image' in obs:
            frame = obs['agentview_image']
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)
        
        if done:
            break
    
    print(f"Episode {episode+1}: reward = {episode_reward:.2f}")
    wandb.log({"episode_reward": episode_reward})

out.release()
env.close()
print(f"Video saved to {video_file}")

# Log video to wandb
wandb.log({"eval_video": wandb.Video(video_file, fps=30)})
wandb.finish()
print("Video logged to wandb!")
