"""
Video recording using agentview camera with EGL
"""
import os
import sys
sys.path.insert(0, '/teamspace/studios/this_studio/LIBERO')

import numpy as np
import cv2

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import DenseRewardEnv

# Create env with camera and EGL
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
env.reset()

# Video path
video_path = "/teamspace/studios/this_studio/cleanrl/runs/libero_spatial_task0__ppo_fast_libero__1__eval/videos/"
os.makedirs(video_path, exist_ok=True)
video_file = os.path.join(video_path, "eval.mp4")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_file, fourcc, 30, (640, 480))

print(f"Saving video to {video_file}")

for episode in range(1):
    env.reset()
    for step in range(300):
        # Random action
        ctrl = env.sim.model.actuator_ctrlrange[:7]
        action = np.random.uniform(ctrl[:, 0], ctrl[:, 1])
        
        # Step
        obs, reward, done, info = env.step(action)
        
        # Get image from agentview
        if 'agentview_image' in obs:
            frame = obs['agentview_image']
            # Robosuite returns RGB, OpenCV needs BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)
        
        if done:
            break
    
    print(f"Episode {episode+1} done")

out.release()
env.close()
print(f"Video saved to {video_file}")
