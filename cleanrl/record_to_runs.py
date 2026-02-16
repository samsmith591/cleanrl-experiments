"""
Simple video recording using mujoco's built-in renderer - saves to runs folder
"""
import os
import sys
sys.path.insert(0, '/teamspace/studios/this_studio/LIBERO')

import numpy as np
import mujoco
import cv2
import glob

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

# Create offscreen renderer with no context
try:
    # Try using mujoco.viewer or direct rendering
    width, height = 640, 480
    
    # Use the sim's renderer if available
    if hasattr(env.sim, 'renderer') and env.sim.renderer is not None:
        renderer = env.sim.renderer
    else:
        # Create new renderer
        renderer = mujoco.Renderer(model)
        renderer.offscreen = True
    
    # Video path in runs folder
    run_name = "libero_spatial_task0__ppo_fast_libero__1__test"
    video_path = f"/teamspace/studios/this_studio/cleanrl/runs/{run_name}/videos/"
    os.makedirs(video_path, exist_ok=True)
    video_file = os.path.join(video_path, "eval.mp4")
    
    out = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
    
    print(f"Saving video to {video_file}")
    
    for episode in range(1):
        env.reset()
        for step in range(300):
            # Random action
            ctrl = env.sim.model.actuator_ctrlrange[:7]
            action = np.random.uniform(ctrl[:, 0], ctrl[:, 1])
            
            # Step
            env.step(action)
            
            # Try to render
            try:
                # Use sim.render for offscreen
                frame = env.sim.render(width, height, camera_id=0, mode='offscreen')
                if frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame)
            except Exception as e:
                print(f"Render error: {e}")
                break
        
        print(f"Episode {episode+1} done")

    out.release()
    print(f"Video saved to {video_file}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

env.close()
