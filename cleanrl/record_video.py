"""
Record video of trained PPO policy on LIBERO using offscreen renderer
"""
import os
import sys
import time

sys.path.insert(0, '/teamspace/studios/this_studio')
sys.path.insert(0, '/teamspace/studios/this_studio/cleanrl')
sys.path.insert(0, '/teamspace/studios/this_studio/LIBERO')

import numpy as np
import torch
import cv2

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import DenseRewardEnv


class ActorCritic(torch.nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims=[256, 256]):
        super().__init__()
        
        policy_layers = []
        in_dim = obs_dim
        for h_dim in hidden_dims:
            policy_layers.extend([
                torch.nn.Linear(in_dim, h_dim),
                torch.nn.Tanh()
            ])
            in_dim = h_dim
        self.policy = torch.nn.Sequential(*policy_layers, torch.nn.Linear(in_dim, action_dim))
        
        self.log_std = torch.nn.Parameter(torch.zeros(action_dim))
        
    def get_eval_action(self, x):
        return self.policy(x)


def record_video(model_path=None, num_episodes=1, save_path='/teamspace/studios/this_studio/libero_video.mp4'):
    device = torch.device("cpu")
    
    # Create env with offscreen rendering
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict['libero_spatial']()
    task = task_suite.get_task(0)
    task_bddl_file = os.path.join(get_libero_path('bddl_files'), task.problem_folder, task.bddl_file)
    
    env = DenseRewardEnv(
        bddl_file_name=task_bddl_file,
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera='frontview',
        camera_heights=480,
        camera_widths=640
    )
    
    obs_dim = 26
    action_dim = 7
    
    # Model
    model = ActorCritic(obs_dim, action_dim).to(device)
    
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print("No model found, using random policy")
    
    model.eval()
    
    # Get action dim from env
    ctrlrange = env.sim.model.actuator_ctrlrange[:7]
    
    # Video writer
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, 30.0, (width, height))
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        
        for step in range(300):
            # Extract state features (same as training)
            proprio = obs.get('robot0_proprio-state', np.zeros(14))
            obj_state = obs.get('object-state', np.zeros(12))
            goal = np.array([0.0, 0.2, 0.0])
            
            obs_state = np.concatenate([proprio, obj_state[:6], goal]).astype(np.float32)[:26]
            if len(obs_state) < 26:
                obs_state = np.pad(obs_state, (0, 26 - len(obs_state)))
            
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs_state).to(device)
                action = model.get_eval_action(obs_tensor)
                action_np = action.cpu().numpy()
            
            # Scale action
            action_scaled = np.clip(action_np, -1, 1)
            action_muj = (action_scaled + 1) / 2 * (ctrlrange[:, 1] - ctrlrange[:, 0]) + ctrlrange[:, 0]
            
            # Step
            obs, reward, done, info = env.step(action_muj)
            episode_reward += reward
            
            # Render offscreen
            frame = env.render(mode='rgb_array', camera_name='frontview')
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame)
            
            if done:
                break
        
        print(f"Episode {episode + 1}: reward = {episode_reward:.2f}")
    
    out.release()
    env.close()
    print(f"Video saved to {save_path}")
    return save_path


if __name__ == "__main__":
    record_video(num_episodes=1)
