"""
Simple video recording using mujoco's built-in renderer
"""
import os
import sys
sys.path.insert(0, '/teamspace/studios/this_studio/LIBERO')

import numpy as np
import mujoco
import cv2

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import DenseRewardEnv

# Create env
benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict['libero_spatial']()
task = task_suite.get_task(0)
task_bddl_file = os.path.join(get_libero_path('bddl_files'), task.problem_folder, task.bddl_file)

env = DenseRewardEnv(bddl_file_name=task_bddl_file, use_camera_obs=False)
env.reset()

# Get MuJoCo model and data
model = env.sim.model
data = env.sim.data

# Create offscreen renderer
renderer = mujoco.Renderer(model, 480, 640)

# Video
out = cv2.VideoWriter('/teamspace/studios/this_studio/libero_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))

for episode in range(1):
    env.reset()
    for step in range(300):
        # Random action
        ctrl = env.sim.model.actuator_ctrlrange[:7]
        action = np.random.uniform(ctrl[:, 0], ctrl[:, 1])
        
        # Step
        env.step(action)
        
        # Render
        renderer.update_scene(data, camera='frontview')
        frame = renderer.render()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
        
    print(f"Episode {episode+1} done")

out.release()
env.close()
print("Video saved!")
