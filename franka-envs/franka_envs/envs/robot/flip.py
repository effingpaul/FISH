from cgitb import enable
import time
from turtle import home
import gym
from gym import spaces
from franka_envs.envs import franka_env
from franka_envs.utils import quaternion_multiplication
import cv2

import numpy as np

class FrankaFlipEnv(franka_env.FrankaEnv):
	def __init__(self, height=84, width=84, step_size=10, enable_arm=True, enable_gripper=True, enable_camera=True, camera_view='side',
				 use_depth=False, dist_threshold=0.05, random_start=True, x_limit=None, y_limit=None, z_limit=None, pitch=0, roll=136, keep_gripper_closed=True, goto_zero_at_init=False, use_real_robot=False, execute_step_wise=True):
		franka_env.FrankaEnv.__init__(
			self,
			home_displacement=[1.7, 1.20, 1.82],
			height=height,
			width=width,
			step_size=step_size,
			enable_arm=enable_arm, 
			enable_gripper=enable_gripper,
			enable_camera=enable_camera,
			camera_view=camera_view,
			use_depth=use_depth,
			keep_gripper_closed=keep_gripper_closed,
			highest_start=True,
			x_limit=x_limit,
			y_limit=y_limit,
			z_limit=z_limit,
			pitch=pitch,
			roll=roll,
			goto_zero_at_init=goto_zero_at_init,
			use_real_robot=use_real_robot,
			execute_step_wise=execute_step_wise
		)
		self.action_space = spaces.Box(low = np.array([-1,-1,-1, -1, -1, -1, -1],dtype=np.float32), 
									   high = np.array([1, 1, 1, 1, 1, 1, 1],dtype=np.float32),
									   dtype = np.float32)
		self.dist_threshold = dist_threshold
		self.random_start = random_start

		if self.random_start:
			self.random_limits = [
				[self.x_limit[0], self.x_limit[1]],
				[self.y_limit[0], self.y_limit[1]],
				[self.z_limit[0], self.z_limit[1]]
			]
		self.z_pos_limit = 1.1

		# Set limits for actions based on max and min in colleced data
		# NOTE: replaced with franka numbers



		self.amin = np.array([-0.005417199999999997, -0.0043680000000000385, -0.005689999999999973, -5.066925490872517 ,-2.1016945376923317 ,-6.599694925074768  ])
		self.amax = np.array([0.0132467, 0.00965499999999997, 0.002778000000000058, 0.9899210406374692, 2.902934758408099, 2.6491223307038965  ])

		self.posmin = np.array([-0.143946, 0.275992, 0.912367, -15.931071325442407, -27.144403233550914, -12.343664422404952 ])
		self.posmax = np.array([0.0940041, 0.532463, 1.29168, 70.79504930395238, 81.03868707608807, 71.8365138061432  ])

		self.anglemin = np.array([-15.931071325442407, -27.144403233550914, -12.343664422404952 ])
		self.anglemax = np.array([70.79504930395238, 81.03868707608807, 71.8365138061432  ])


		self.diff = self.amax - self.amin
		self.posdiff = self.posmax - self.posmin

	def arm_refresh(self, reset=True):
		self.arm.clear_errors()
		self.arm.set_mode_and_state()
		if reset:
			self.arm.reset(home=True, reset_at_home=True)
		time.sleep(2)

	def reset(self):
		if not self.enable_arm:
			return np.array([0,0,0], dtype=np.float32)
		self.arm_refresh(reset=True)
		
		if self.keep_gripper_closed:
			self.arm.close_gripper_fully()
		else:
			self.arm.open_gripper_fully()

		time.sleep(0.4)		
		obs = {}
		obs['features'] = np.array(self.arm.get_position(), dtype=np.float32)
		obs['pixels'] = self.render(mode='rgb_array', width=self.width, height=self.height)
		return obs

	def get_random_pos(self, pos=None):
		x_disp = self.random_limits[0][0] + np.random.rand()*(self.random_limits[0][1] - self.random_limits[0][0])
		y_disp = self.random_limits[1][0] + np.random.rand()*(self.random_limits[1][1] - self.random_limits[1][0])
		z_disp = self.random_limits[2][0] + np.random.rand()*(self.random_limits[2][1] - self.random_limits[2][0])
		
		if pos is None:
			pos = np.zeros(3).astype(np.float32)
		pos[0] = x_disp
		pos[1] = y_disp 
		pos[2] = z_disp if not self.arm.highest_start else self.z_limit[1] 									
		return pos

	def set_random_pos(self):
		self.arm.clear_errors()
		self.arm.set_mode_and_state()
		pos = self.arm.get_position()
		
		# Move up
		pos[2] = self.arm.z_limit[1]
		self.arm.set_position(pos)

		# Set random pos
		pos = self.get_random_pos(pos)								

		self.arm.set_position(pos)
		if self.arm.keep_gripper_closed:
			self.arm.close_gripper_fully()
		else:
			self.arm.open_gripper_fully()


	def step(self, action):
		if self.arm.has_error():
			self.arm.clear_errors()
			self.arm.set_mode_and_state()

		# TODO: these three have to be chked for necessaity and then moved to the config
		use_quaternion = True
		tau = 0.25
		use_fixed_move_time = True

		new_pos = self.arm.get_position(use_quaternion=use_quaternion)
		print("current pos: ", new_pos)
		print("action: ", action)

		action = action # * self.posdiff # + self.amin
		print("action upscaled: ", action)

		# to treplay the expert trajectory
		# action = self.get_expert_label(self.step_number) # * self.posdiff
		# print("expert action: ", action)

		if use_quaternion:
			for i in range(3):
				new_pos[i] += action[i]

			new_pos[3:] = quaternion_multiplication(new_pos[3:], action[3:])
		else:
			new_pos += action



		# use this to directlz execute quaternion poses
		exp_action = self.get_expert_label(self.step_number)
		
		print("expert action: ", exp_action)
		# new_pos = action
		

		if self.enable_arm:
			if use_fixed_move_time:
				self.set_position(new_pos, False, use_quaternion, tau)
			else:
				self.set_position(new_pos, False, use_quaternion)
			time.sleep(tau)

		if self.enable_gripper:
			if action[3]>0.5:
				self.arm.open_gripper_fully()
				time.sleep(0.2)
			elif action[3]<-0.5:
				self.arm.close_gripper_fully()
				time.sleep(0.2)

		self.reward = 0
		
		done = False
		
		info = {}
		info['is_success'] = 1 if self.reward==1 else 0

		obs = {}
		obs['features'] = np.array(self.arm.get_position(), dtype=np.float32)
		obs['pixels'] = self.render(mode='rgb_array', width=self.width, height=self.height, step_number=self.step_number)
		self.step_number += 1

		if self.execute_step_wise:
			# display the newly obtained rendered image in a window and wait for user input
			# scale the whole obs array to be an int within 0 and 255 for cv2
			if len(obs['pixels'].shape) == 3:
				obs['pixels'] = np.array(obs['pixels'], dtype=np.uint8)
				cv2.imshow('image', obs['pixels'])
				cv2.waitKey(0)

				cv2.destroyAllWindows()


		return obs, self.reward, done, info

	def set_position(self, pos, wait=False, use_quaternion=False, tau=None):
		# pos = self.limit_pos(pos)
		#x = (pos[0] + self.arm.zero[0])*100
		#y = (pos[1] + self.arm.zero[1])*100
		#z = (pos[2] + self.arm.zero[2])*100
		#self.arm.arm.set_position(x=x, y=y, z=z, roll=pos[3], pitch=pos[4], yaw=pos[5], wait=wait)
		self.arm.set_position(pos=pos, wait=wait, use_pitch=True, use_roll=True, use_yaw=True, use_quaternion=use_quaternion, tau=tau)

	def limit_pos(self, pos):
		if pos[2] <= self.z_pos_limit:
			pos[1] = max(self.random_limits[1][0], pos[1])
			pos[1] = min(self.random_limits[1][1], pos[1])
		else:
			pos[1] = max(self.y_limit[0], pos[1])
			pos[1] = min(self.y_limit[1], pos[1])
		pos[0] = max(self.x_limit[0], pos[0])
		pos[0] = min(self.x_limit[1], pos[0])
		pos[1] = max(self.y_limit[0], pos[1])
		pos[1] = min(self.y_limit[1], pos[1])
		pos[2] = max(self.z_limit[0], pos[2])
		pos[2] = min(self.z_limit[1], pos[2])

		# limit pitch roll yaw angles to be within range in degrees

		pos[3] = min(max(self.anglemin[0], pos[3]), self.anglemax[0])
		pos[4] = min(max(self.anglemin[1], pos[4]), self.anglemax[1])
		pos[5] = min(max(self.anglemin[2], pos[5]), self.anglemax[2])
		

		#pos[3] = min(max(110, pos[3]), 136)
		#pos[4] = min(max(-2, pos[4]), 2)
		#pos[5] = min(max(-2, pos[5]), 2)

		return pos