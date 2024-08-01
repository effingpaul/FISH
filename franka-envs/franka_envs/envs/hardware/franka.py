import sys
import time
import numpy as np
import random
from configparser import ConfigParser
from franka_envs.utils import quatPoseToPRYPose, PRYPoseToQuatPose
from franka_envs.envs.hardware.franka_request import FrankaRequestNode
import robotic as ry


class Franka:

	def __init__(self, config_file='./franka.conf', home_displacement = (0,0.276,1.266), low_range=(0.2,0.2,0.1) , high_range=(0.4,0.4,0.2),
				 keep_gripper_closed=False, highest_start=False, x_limit=None, y_limit=None, z_limit=None, yaw_limit=None,
				 pitch = 0, roll=180, yaw=0, gripper_action_scale=200, start_at_the_back=False, use_real_robot=False, execute_step_wise=True, qHome = None,
				 use_external_robot_system = False, external_ip_address = "0.0.0.0",):
		self.gripper_max_open = 800
		self.gripper_min_open = 0
		self.zero = (206/100,0/100,120.5/100)	# Units: .1 meters 
		self.home = home_displacement
		self.keep_gripper_closed = keep_gripper_closed
		self.highest_start = highest_start
		self.low_range = low_range
		self.high_range = high_range
		self.joint_limits = None
		self.gripper_action_scale = gripper_action_scale
		self.start_at_the_back = start_at_the_back
		self.use_real_robot = use_real_robot
		self.execute_step_wise = execute_step_wise
		self.use_external_robot_system=use_external_robot_system
		self.external_ip_address=external_ip_address

		self.qHome = qHome

		# Limits
		self.x_limit = [-0.3, 0.3] if x_limit is None else x_limit
		self.y_limit = [0.0, 0.7] if y_limit is None else y_limit
		self.z_limit = [0.8, 1.4] if z_limit is None else z_limit
		self.yaw_limit = None if yaw_limit is None else yaw_limit 

		# Pitch value - Horizontal or vertical orientation
		self.pitch = pitch
		self.roll = roll
		self.yaw = yaw


	def start_robot(self):

		if self.use_external_robot_system:
			self.external_bot = FrankaRequestNode(self.external_ip_address)
			return
		else:
			# Create Configuration
			self.C = ry.Config()
			self.C.addFile(ry.raiPath('scenarios/pandaSingle.g'))
			self.C.addFrame('target')
			self.C.view(False)
			self.bot = ry.BotOp(self.C, useRealRobot=self.use_real_robot)
			if self.qHome is None:
				self.qHome = self.C.getJointState()

			if self.execute_step_wise:
				# Create Dummy Configuration
				self.simC = ry.Config()
				self.simC.addFile(ry.raiPath('scenarios/pandaSingle.g'))
				self.simC.view(False)
				self.simBot = ry.BotOp(self.simC, useRealRobot=False)

			#if self.arm.error_code != 0:
			#	self.arm.clean_error()
			#self.set_mode_and_state()

	def set_mode_and_state(self, mode=0, state=0):
		# TODO
		return
		#self.arm.set_mode(mode)
		#self.arm.set_state(state=state)

	def clear_errors(self):
		# TODO
		return
		#self.arm.clean_warn()
		#self.arm.clean_error()

	def clear(self):
		if not self.use_external_robot_system:
			del self.C
			del self.bot

		if self.execute_step_wise:
			del self.simBot
			del self.simC

	def has_error(self):
		return False

	def reset(self, home = False, reset_at_home=True):
		#if self.arm.has_err_warn:
		#	self.clear_errors()
		if home:
			if reset_at_home:
				self.move_to_home()
			else:
				self.move_to_zero()
			if self.keep_gripper_closed:
				self.close_gripper_fully()
			else:
				self.open_gripper_fully()

	def move_to_home(self, open_gripper=False):

		if self.use_external_robot_system:
			self.external_bot.send_home_command(self.qHome)
			return
		else:
			# move to the home position specified by qHome
			self.bot.moveTo(self.qHome, 0.2, True)

			while (self.bot.getTimeToEnd()>0.) :
				self.bot.sync(self.C,0)
			self.bot.sync(self.C, 0)

			if open_gripper and not self.keep_gripper_closed:
				self.open_gripper_fully()

			print("actual home pose", self.C.getFrame("l_gripper").getPosition(), self.C.getFrame("l_gripper").getQuaternion())
			# this shoudl be home : [0.00197139, 0.275992, 1.26491] [-0.77925, -0.199869, 0.144179, -0.576224]

			# print("no home rn")
			# pos = self.get_position()
			# pos[0] = self.home[0]
			# pos[1] = self.home[1]
			# pos[2] = self.home[2]
			# self.set_position(pos)
			# if open_gripper and not self.keep_gripper_closed:
			# 	self.open_gripper_fully()

	def IK(self, C, pos, use_quaternion=False):
		# set target to pos and quat
		if not use_quaternion:
			pos = PRYPoseToQuatPose(pos)
		print("optimized pos: ", pos)
		self.C.getFrame('target').setPosition(pos[:3]).setQuaternion(pos[3:])

		# create komo problem to calculate joint configuration
		q0 = C.getJointState()
		komo = ry.KOMO(self.C, 1, 1, 0, False) #one phase one time slice problem, with 'delta_t=1', order=0
		komo.addObjective([], ry.FS.jointState, [], ry.OT.sos, [1e-1], q0) #cost: close to 'current state'
		komo.addObjective([], ry.FS.jointState, [], ry.OT.sos, [1e-1], self.qHome) #cost: close to qHome
		komo.addObjective([1.], ry.FS.positionDiff, ['l_gripper', 'target'], ry.OT.eq, [1e1]) #constraint: gripper position
		komo.addObjective([1.], ry.FS.quaternionDiff, ['l_gripper', 'target'], ry.OT.eq, [1e3]) # contraint: gripper orientation
		
		ret = ry.NLP_Solver(komo.nlp(), verbose=0).solve()
		return komo.getPath()[0] #[komo.getPath()[0], ret]


	def set_random_pos(self):
		self.clear_errors()
		self.set_mode_and_state()
		pos = self.get_position()
		
		# Move up
		pos[2] = self.z_limit[1]
		self.set_position(pos)

		# Set random pos
		x_disp = self.low_range[0] + np.random.rand()*(self.high_range[0] - self.low_range[0])
		y_disp = self.low_range[1] + np.random.rand()*(self.high_range[1] - self.low_range[1])
		z_disp = self.low_range[2] + np.random.rand()*(self.high_range[2] - self.low_range[2])
		
		pos[0] = self.home[0] + x_disp * np.random.choice([-1,1])		# Here we sample in a square ring around the home 
		pos[1] = self.home[1] + y_disp * np.random.choice([-1,1])		# Here we sample in a square ring around the home 
		pos[2] = self.home[2] + z_disp if not self.highest_start else self.z_limit[1] 									# For z we jsut sample from [a,b]

		self.set_position(pos)
		if self.keep_gripper_closed:
			self.close_gripper_fully()
		else:
			self.open_gripper_fully()

	def move_to_zero(self):
		pos = self.get_position()
		pos[0] = min(max(self.x_limit[0],0), self.x_limit[1])# 0
		pos[1] = min(max(self.y_limit[0],0), self.y_limit[1])# 0
		pos[2] = min(max(self.z_limit[0],0), self.z_limit[1]) if not self.highest_start else self.z_limit[1] # 0
		self.set_position(pos)

	def set_position(self, pos, wait=False, use_roll=False, use_pitch=False, use_yaw=False, use_quaternion=False, tau=None):

		pos = self.limit_pos(pos)
		# x = (pos[0] + self.zero[0])*100
		# y = (pos[1] + self.zero[1])*100
		# z = (pos[2] + self.zero[2])*100
		roll = pos[3] if use_roll else self.roll
		pitch = pos[4] if use_pitch else self.pitch
		yaw = pos[5] if use_yaw else self.yaw
		print("desired pos: ", pos)

		if self.use_external_robot_system:
			self.external_bot.send_set_position_command(pos)
			return
		else:
			q = self.IK(self.C, pos, use_quaternion=use_quaternion)

			# if step wise execution is on we have to show the movement in a simulation window first
			if self.execute_step_wise:

				# move to position
				self.simBot.moveTo(q, 2., True)
				self.simBot.sync(self.C, 0)
				# wait for user input
				input("Press Enter to continue...")


			if tau is None:
				self.bot.moveTo(q, 1., True)
			else:
				self.bot.move([q], [tau], False) #True
			self.bot.sync(self.C, 0)

			#self.arm.set_position(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw, wait=wait)

	def get_position(self, use_quaternion=False):

		if self.use_external_robot_system:
			pos = self.external_bot.send_get_position_command()
			print("pos received form external bot: ", pos)
		else:

			self.bot.sync(self.C, 0)
			#pos = self.arm.get_position()[1]
			position = self.C.getFrame('l_gripper').getPosition()
			quaternion = self.C.getFrame('l_gripper').getQuaternion()
			pos = [position[0], position[1], position[2], quaternion[0], quaternion[1], quaternion[2], quaternion[3]]

		if use_quaternion:
			return pos
		pos = quatPoseToPRYPose(pos)
		# x = (pos[0]/100.0 - self.zero[0])
		# y = (pos[1]/100.0 - self.zero[1])
		# z = (pos[2]/100.0 - self.zero[2])
		return np.array([pos[0], pos[1], pos[2], pos[3], pos[4], pos[5]]).astype(np.float32)

	def get_gripper_position(self):
		# TODO
		return 0

	def open_gripper_fully(self):
		if self.use_external_robot_system:
			self.external_bot.send_open_gripper_fully_command()
			return
		else:
			#self.set_gripper_position(self.gripper_max_open)
			self.bot.gripperMove(ry._left, width=.07)


	def close_gripper_fully(self):
		if self.use_external_robot_system:
			self.external_bot.send_close_gripper_fully_command()
			return
		else:
			#self.set_gripper_position(self.gripper_min_open)
			self.bot.gripperMove(ry._left, width=.003)

	def open_gripper(self):
		# TODO
		return
		#self.set_gripper_position(self.get_gripper_position() + self.gripper_action_scale)

	def close_gripper(self):
		# TODO
		return
		#self.set_gripper_position(self.get_gripper_position() - self.gripper_action_scale)

	def set_gripper_position(self, pos, wait=False):
		'''
		wait: To wait till completion of action or not
		'''
		# TODO
		return

	def get_servo_angle(self):
		angles = self.C.getJointState()
		return angles

	def set_servo_angle(self, angles, is_radian=None):
		'''
		angles: List of length 6
		'''
		self.bot.moveTo(angles, {1.}, True)
		self.bot.sync(self.C, 0)
	
	def limit_pos(self, pos):
		pos[0] = max(self.x_limit[0], pos[0])
		pos[0] = min(self.x_limit[1], pos[0])
		pos[1] = max(self.y_limit[0], pos[1])
		pos[1] = min(self.y_limit[1], pos[1])
		pos[2] = max(self.z_limit[0], pos[2])
		pos[2] = min(self.z_limit[1], pos[2])
		if self.yaw_limit is not None:
			pos[5] = max(self.yaw_limit[0], pos[5])
			pos[5] = min(self.yaw_limit[1], pos[5])
		return pos
	
	def get_image(self):
		if self.use_external_robot_system:
			#TODO
			return
		else:
			image, depth, points = self.bot.getImageDepthPcl("l_gripper")
			return image