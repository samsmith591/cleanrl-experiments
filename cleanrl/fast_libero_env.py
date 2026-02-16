"""
Fast LIBERO environment - bypasses Robosuite wrapper for ~6000 SPS
"""
import numpy as np
import gymnasium as gym
import os
import sys

# Add LIBERO path
sys.path.insert(0, '/teamspace/studios/this_studio/LIBERO')

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import DenseRewardEnv


class FastLIBEROEnv:
    """
    Fast LIBERO using raw MuJoCo for stepping.
    Observation: robot proprio + simple object positions.
    Reward: dense distance-based reward.
    """
    
    def __init__(self, benchmark_name='libero_spatial', task_id=0, max_episode_steps=300, num_sim_steps=1):
        self.benchmark_name = benchmark_name
        self.task_id = task_id
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        
        # Get task
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[benchmark_name]()
        task = task_suite.get_task(task_id)
        task_bddl_file = os.path.join(get_libero_path('bddl_files'), task.problem_folder, task.bddl_file)
        
        # Create base env
        self.env = DenseRewardEnv(
            bddl_file_name=task_bddl_file,
            use_camera_obs=False,
            has_offscreen_renderer=False,
            has_renderer=False
        )
        
        # Action space: 7DOF arm
        self._action_dim = 7
        self._num_sim_steps = num_sim_steps
        
        # Get robot and object info
        self._setup()
        
        # Observation space: robot proprio (14) + object positions (6 per object)
        self.obs_dim = 14 + 6 * len(self.target_objects)
        self.observation_space = gym.spaces.Box(
            low=-10, high=10, shape=(self.obs_dim,), dtype=np.float32
        )
        
        # Action space: 7DOF arm + 1 gripper (use 7 for now)
        self._action_dim = 7
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(self._action_dim,), dtype=np.float32
        )
        
    def _setup(self):
        """Extract object names and control range"""
        # Get inner env
        inner = self.env.env if hasattr(self.env, 'env') else self.env
        
        # Get robot
        self.robot = inner.robots[0]
        
        # Get target objects (first 2)
        self.target_objects = []
        for name in inner.objects_dict.keys():
            if len(self.target_objects) >= 2:
                break
            self.target_objects.append(name)
        
        # Get goal position (from task)
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[self.benchmark_name]()
        task = task_suite.get_task(self.task_id)
        self.goal_description = task.language
        
        # Control range
        self.ctrlrange = self.env.sim.model.actuator_ctrlrange[:self._action_dim]
        
    def _get_obs(self):
        """Fast observation: robot proprio + object positions"""
        sim = self.env.sim
        
        # Robot proprioception (qpos + qvel) - use sim directly
        qpos = sim.data.qpos[:self._action_dim]
        qvel = sim.data.qvel[:self._action_dim]
        proprio = np.concatenate([qpos, qvel])
        
        # Object positions (just xyz)
        obj_pos = []
        for name in self.target_objects:
            try:
                body_id = sim.model.body_name2id(name)
                pos = sim.data.body_xpos[body_id]
                obj_pos.append(pos[:3])
            except:
                obj_pos.append(np.zeros(3))
        
        # Goal position (approximate - use table center)
        goal_pos = np.array([0.0, 0.2, 0.0])  # Approximate goal location
        
        # Combine
        obs = np.concatenate([proprio] + obj_pos + [goal_pos]).astype(np.float32)
        
        # Pad if needed
        if len(obs) < self.obs_dim:
            obs = np.pad(obs, (0, self.obs_dim - len(obs)))
        elif len(obs) > self.obs_dim:
            obs = obs[:self.obs_dim]
            
        return obs
    
    def _compute_reward(self, action):
        """Dense reward: distance to goal + action penalty"""
        sim = self.env.sim
        
        # Get gripper position
        gripper_pos = sim.data.body_xpos[self.robot.eef_site_id]
        
        # Get first object position
        try:
            obj_pos = sim.data.body_xpos[
                sim.model.body_name2id(self.target_objects[0])
            ]
        except:
            obj_pos = np.zeros(3)
        
        # Distance reward (negative distance)
        dist = np.linalg.norm(gripper_pos[:2] - obj_pos[:2])  # 2D distance
        
        # Height reward (lift object)
        height_reward = max(0, obj_pos[2] - 0.05) * 10
        
        # Action penalty
        action_penalty = -0.01 * np.sum(np.square(action))
        
        reward = -dist + height_reward + action_penalty
        return reward
    
    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.env.reset()
        
        # Initialize ctrl
        self.env.sim.data.ctrl[:] = 0
        return self._get_obs(), {}
    
    def step(self, action):
        # Scale action from [-1, 1] to control range
        action = np.clip(action, -1, 1)
        action_scaled = (action + 1) / 2 * (self.ctrlrange[:, 1] - self.ctrlrange[:, 0]) + self.ctrlrange[:, 0]
        
        # Apply action directly to MuJoCo
        self.env.sim.data.ctrl[:self._action_dim] = action_scaled
        
        # Step physics - configurable number of sim steps per action
        for _ in range(self._num_sim_steps):
            self.env.sim.step()
        
        # Compute reward
        reward = self._compute_reward(action)
        
        # Check termination
        self.current_step += 1
        
        # Simple termination: max steps
        done = False
        truncated = self.current_step >= self.max_episode_steps
        if truncated:
            done = True
            
        info = {}
        
        return self._get_obs(), reward, done, truncated, info
    
    def close(self):
        self.env.close()
    
    @property
    def unwrapped(self):
        return self


if __name__ == '__main__':
    import time
    
    env = FastLIBEROEnv()
    obs, _ = env.reset()
    print(f"Obs shape: {obs.shape}")
    print(f"Action shape: {env.action_space.shape}")
    
    # Test speed
    start = time.time()
    episodes = 0
    steps = 0
    for i in range(500):
        action = env.action_space.sample()
        obs, reward, done, truncated, _ = env.step(action)
        steps += 1
        if done:
            env.reset()
            episodes += 1
    elapsed = time.time() - start
    
    print(f"FastLIBERO: {steps/elapsed:.0f} SPS, {episodes} episodes")
    env.close()
