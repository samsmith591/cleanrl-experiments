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
    
    def __init__(self, benchmark_name='libero_spatial', task_id=0, max_episode_steps=300, num_sim_steps=1, render_mode=None):
        self.benchmark_name = benchmark_name
        self.task_id = task_id
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        self.render_mode = render_mode
        
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
        # Observation space: joint positions (7) + relative object positions (6 for 2 objects)
        self.obs_dim = 7 + 6 * len(self.target_objects)  # 7 + 6*2 = 19
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
        """Observation: robot pose + object positions relative to gripper"""
        sim = self.env.sim
        
        # Robot joint positions (7)
        qpos = sim.data.qpos[:self._action_dim]
        
        # Object positions relative to gripper
        gripper_pos = sim.data.body_xpos[self.robot.eef_site_id]
        
        rel_obj_pos = []
        for name in self.target_objects:
            try:
                body_id = sim.model.body_name2id(name)
                obj_pos = sim.data.body_xpos[body_id]
                # Relative position: object - gripper
                rel_pos = obj_pos - gripper_pos
                rel_obj_pos.append(rel_pos)
            except:
                rel_obj_pos.append(np.zeros(3))
        
        # Combine: joint_pos (7) + relative positions (6 for 2 objects)
        obs = np.concatenate([qpos] + rel_obj_pos).astype(np.float32)
        
        # Pad to fixed size
        if len(obs) < self.obs_dim:
            obs = np.pad(obs, (0, self.obs_dim - len(obs)))
        elif len(obs) > self.obs_dim:
            obs = obs[:self.obs_dim]
            
        return obs
    
    def _compute_reward(self, action):
        """Dense reward: returns total and components"""
        sim = self.env.sim
        
        # Get gripper position
        gripper_pos = sim.data.body_xpos[self.robot.eef_site_id]
        
        # Get object positions (assume first is bowl, second is plate)
        try:
            bowl_pos = sim.data.body_xpos[
                sim.model.body_name2id(self.target_objects[0])
            ]
        except:
            bowl_pos = np.zeros(3)
        
        try:
            plate_pos = sim.data.body_xpos[
                sim.model.body_name2id(self.target_objects[1])
            ]
        except:
            plate_pos = np.zeros(3)
        
        # Distance 1: gripper to bowl (exp(-dist^2 * 5))
        dist_gripper_bowl = np.linalg.norm(gripper_pos - bowl_pos)
        reward_gripper_bowl = np.exp(-dist_gripper_bowl * dist_gripper_bowl * 5.0)
        
        # Distance 2: bowl to plate - normalized by initial distance
        # Reward is high (close to 1) only when bowl is CLOSER than initial
        dist_bowl_plate = np.linalg.norm(bowl_pos - plate_pos)
        if hasattr(self, 'init_bowl_plate_dist') and self.init_bowl_plate_dist > 0.001:
            # Normalize: positive reward if closer than initial, negative if farther
            normalized_dist = dist_bowl_plate / self.init_bowl_plate_dist
            reward_bowl_plate = np.exp(-(normalized_dist - 1.0)**2 * 5.0)  # max at 1.0 (initial dist)
        else:
            reward_bowl_plate = 0
        
        # Height reward: reward for lifting bowl
        height_reward = max(0, bowl_pos[2] - 0.05)
        
        reward = reward_gripper_bowl + reward_bowl_plate + height_reward
        
        # Return total and components
        return reward, {
            'reward_gripper_bowl': reward_gripper_bowl,
            'reward_bowl_plate': reward_bowl_plate,
            'reward_height': height_reward,
        }
    
        # Get initial bowl-plate distance at reset
        self._compute_init_distance()
        
        return self._get_obs(), {}
    
    def _compute_init_distance(self):
        """Compute initial distance between bowl and plate"""
        sim = self.env.sim
        try:
            init_bowl = sim.data.body_xpos[sim.model.body_name2id(self.target_objects[0])]
            init_plate = sim.data.body_xpos[sim.model.body_name2id(self.target_objects[1])]
            self.init_bowl_plate_dist = np.linalg.norm(init_bowl - init_plate)
        except:
            self.init_bowl_plate_dist = 0.1  # default
    
    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.env.reset()
        
        # Initialize ctrl
        self.env.sim.data.ctrl[:] = 0
        
        # Compute initial distance
        self._compute_init_distance()
        
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
        
        # Compute reward (returns total and components)
        reward, reward_info = self._compute_reward(action)
        
        # Check termination
        self.current_step += 1
        
        # Simple termination: max steps
        done = False
        truncated = self.current_step >= self.max_episode_steps
        if truncated:
            done = True
            
        info = reward_info
        
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
