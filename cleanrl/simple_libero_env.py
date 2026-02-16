"""
Simplified LIBERO environment - uses raw MuJoCo state instead of computed observations
"""
import numpy as np
import gymnasium as gym
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import DenseRewardEnv
import os


class SimpleLIBEROEnv:
    """Simplified LIBERO with minimal observation computation"""
    
    def __init__(self, benchmark_name='libero_spatial', task_id=0, use_camera_obs=False):
        self.benchmark_name = benchmark_name
        self.task_id = task_id
        
        # Get task
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[benchmark_name]()
        task = task_suite.get_task(task_id)
        task_bddl_file = os.path.join(get_libero_path('bddl_files'), task.problem_folder, task.bddl_file)
        
        # Create base env (without camera for speed)
        self.env = DenseRewardEnv(
            bddl_file_name=task_bddl_file,
            use_camera_obs=False,
            has_offscreen_renderer=False,
            has_renderer=False
        )
        
        # Get dimensions
        self.action_dim = 7  # 7 DOF arm
        self.max_episode_steps = 300
        self.current_step = 0
        
        # Define observation space: robot proprio (14) + object positions (6 per object)
        # Keep it simple: just robot state + 2 object positions
        self.obs_dim = 14 + 12  # proprio (7 joint pos + 7 joint vel) + 2 objects (6D each)
        self.observation_space = gym.spaces.Box(
            low=-10, high=10, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(self.action_dim,), dtype=np.float32
        )
        
        # Get initial state
        self._get_object_names()
        
    def _get_object_names(self):
        """Get names of objects in the scene"""
        # Get objects from env - need to access inner env
        inner_env = self.env.env if hasattr(self.env, 'env') else self.env
        self.object_names = []
        for name in inner_env.objects_dict.keys():
            if 'bowl' in name.lower() or 'plate' in name.lower():
                self.object_names.append(name)
        self.object_names = self.object_names[:2]  # Keep first 2
        
    def _get_obs(self):
        """Fast observation: just robot state + raw object positions"""
        # Robot proprioception (qpos + qvel)
        robot = self.env.robots[0]
        qpos = robot._joint_positions
        qvel = robot._joint_velocities
        proprio = np.concatenate([qpos, qvel])
        
        # Simple object positions (just xyz from sim)
        obj_pos = []
        for name in self.object_names:
            if name in self.env.sim.data.body_xpos:
                pos = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id(name)]
                obj_pos.append(pos[:3])  # Just xyz
            else:
                obj_pos.append(np.zeros(3))
        
        # Flatten
        obj_pos = np.concatenate(obj_pos) if obj_pos else np.zeros(6)
        
        # Combine
        obs = np.concatenate([proprio, obj_pos]).astype(np.float32)
        return obs
    
    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.env.reset()
        return self._get_obs(), {}
    
    def step(self, action):
        # Scale action to control range
        ctrlrange = self.env.sim.model.actuator_ctrlrange[:self.action_dim]
        action_scaled = np.clip(action, -1, 1)
        action_mujo = (action_scaled + 1) / 2 * (ctrlrange[:, 1] - ctrlrange[:, 0]) + ctrlrange[:, 0]
        
        obs, reward, done, info = self.env.step(action_mujo)
        
        self.current_step += 1
        if self.current_step >= self.max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
            
        return self._get_obs(), reward, done, False, info
    
    def close(self):
        self.env.close()
    
    @property
    def unwrapped(self):
        return self


if __name__ == '__main__':
    import time
    
    env = SimpleLIBEROEnv()
    obs = env.reset()
    
    # Test speed
    start = time.time()
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        if done:
            obs = env.reset()[0]
    elapsed = time.time() - start
    print(f'SimpleLIBERO: {100/elapsed:.0f} SPS')
    env.close()
