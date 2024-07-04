from cgitb import enable
import gym
from gym import spaces
import cv2
import numpy as np
import time
import pickle

from pygame import init

from franka_envs.envs.hardware.camera import Camera
from franka_envs.envs.hardware.franka import Franka

class FrankaEnv(gym.Env):
	def __init__(self, home_displacement=[2,-0.21,.4], height=84, width=84, step_size=10, enable_arm=True, enable_gripper=True, enable_camera=True,
				 camera_view='side', use_depth=False, keep_gripper_closed=False, highest_start=False, 
				 x_limit=None, y_limit=None, z_limit=None, yaw_limit=None, pitch=0, roll=180, yaw=0, goto_zero_at_init=False, start_at_the_back=False,use_real_robot=False, execute_step_wise=True,**kwargs):
		super(FrankaEnv, self).__init__()
		self.height = height
		self.width = width
		self.use_depth = use_depth
		self.step_size = step_size
		self.enable_arm = enable_arm
		self.enable_camera = enable_camera
		self.camera_view = camera_view
		self.home_displacement = np.array(home_displacement, dtype=np.float32)
		self.keep_gripper_closed = keep_gripper_closed
		self.highest_start = highest_start
		self.x_limit = x_limit
		self.y_limit = y_limit
		self.z_limit = z_limit
		self.yaw_limit = yaw_limit
		self.pitch = pitch
		self.roll = roll
		self.yaw = yaw
		self.goto_zero_at_init = goto_zero_at_init
		self.start_at_the_back = start_at_the_back
		self.use_real_robot = use_real_robot
		self.execute_step_wise = execute_step_wise
		self.step_number = 0
		
		if self.use_depth:
			self.n_channels = 4
		else:
			self.n_channels = 3  

		if self.enable_arm:
			self.enable_gripper = enable_gripper
		else:
			self.enable_gripper = False
		self.reward = 0
		

		self.observation_space = spaces.Box(low = np.array([0,0],dtype=np.float32), high = np.array([255,255],dtype=np.float32), dtype = np.float32)

		self.observation = np.zeros((self.n_channels, height, width)).astype(np.uint8)
		
		if self.enable_arm:
			self.init_arm()

		if self.enable_camera:
			print("no external camera needed")
			import cv2

			self.cap = cv2.VideoCapture(10)

			# Realsense Camera
			# self.cam = Camera(width=self.width, height=self.height, view=self.camera_view)
		else: 
			# load images from the expert trajectory dataset that is pickled 
			with open("/home/paul/Documents/Robotics_Project/FISH/FISH/expert_demos/frankagym/FrankaFlipBagel-v1/expert_demos.pkl", 'rb') as f:
				self.expertObservations, _, self.expertLabels, _ = pickle.load(f)
				with open("/home/paul/Documents/Robotics_Project/FISH/FISH/expert_demos/frankagym/FrankaFlipBagel-v1/quatPoses.pkl", 'rb') as f:
					self.expertQuatPoses = pickle.load(f)

	
	def init_arm(self):
		self.arm = Franka(home_displacement=self.home_displacement, keep_gripper_closed=self.keep_gripper_closed, highest_start=self.highest_start, 
						x_limit=self.x_limit, y_limit=self.y_limit, z_limit=self.z_limit, yaw_limit=self.yaw_limit, pitch=self.pitch, roll=self.roll,
						yaw=self.yaw, start_at_the_back=self.start_at_the_back, use_real_robot=self.use_real_robot, execute_step_wise=self.execute_step_wise)
		self.arm.start_robot()
		self.arm.clear_errors()
		self.arm.set_mode_and_state()
		self.arm.reset(home=False)
		if self.goto_zero_at_init:
			self.arm.move_to_zero()
		time.sleep(2)

	def step(self, action):
		new_pos = self.arm.get_position()
		new_pos[:3] += action[:3] * 0.25
		gripper_movement = 0
		
		if self.enable_gripper:
			if action[3]>0.5:
				self.arm.open_gripper_fully()
				time.sleep(0.2)
			elif action[3]<-0.5:
				self.arm.close_gripper_fully()
				time.sleep(0.2)

		if self.enable_arm:
			# this is where the movement happens
			self.arm.set_position(new_pos)
			time.sleep(0.4)

		done = False
		
		info = {}
		info['is_success'] = 1 if self.reward==1 else 0

		obs = {}
		obs['features'] = np.array(self.arm.get_position(), dtype=np.float32)
		obs['pixels'] = self.render(mode='rgb_array', width=self.width, height=self.height)

		self.reward = self.get_reward()
		self.step_number += 1
		
		return obs, self.reward, done, info

	def arm_refresh(self, reset=True):
		self.arm.clear_errors()
		self.arm.set_mode_and_state()
		if reset:
			self.arm.reset(home=True)
		time.sleep(2)

	def reset(self):
		if not self.enable_arm:
			return np.array([0,0,0], dtype=np.float32)
		self.arm_refresh(reset=False)
		self.arm.set_random_pos()
		time.sleep(0.4)		
		obs = {}
		obs['features'] = np.array(self.arm.get_position(), dtype=np.float32)
		obs['pixels'] = self.render(mode='rgb_array', width=self.width, height=self.height)

		self.step_number = 0
		return obs

	def render(self, mode='rgb_array', width=84, height=84, step_number=0):
		print("at step: ", step_number)
		if not self.enable_camera:
			# reorgnaize first iage to be in the correct format height x width x channels
			img = np.transpose(self.expertObservations[0][step_number], (1,2,0))
			

			
			return img
		

			#return np.random.rand(height, width, self.n_channels)
		# obs = self.arm.get_image()
		# obs = cv2.resize(obs, (width, height))


		if self.cap.isOpened():
			ret, frame = self.cap.read()
			if ret:
				obs = frame
			
		# Release everything if job is finished
		# self.cap.release()
		
		# obs = self.cam.get_frame()
		#crop image to be quadratic
		if min(obs.shape[0], obs.shape[1]) != obs.shape[0]:
			cutoff = (obs.shape[0] - obs.shape[1]) // 2
			obs = obs[cutoff:-cutoff, :]
		if min(obs.shape[0], obs.shape[1]) != obs.shape[1]:
			cutoff = (obs.shape[1] - obs.shape[0]) // 2
			obs = obs[:, cutoff:-cutoff]
        # rescale images to height and width
		obs = cv2.resize(obs, (width, height))
		return obs[:,:,:self.n_channels]
	
	def get_expert_label(self, step_number):
		return self.expertLabels[0][step_number]

		# use this to give expert quaternion poses
		# return self.expertQuatPoses[step_number]

	def get_reward(self):
		pass
