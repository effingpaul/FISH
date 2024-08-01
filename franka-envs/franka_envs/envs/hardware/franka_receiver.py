import zmq
import pickle
import numpy as np
import robotic as ry
import time
import math


class FrankaController:
    def __init__(self, use_real_robot=True):
        self.use_real_robot = use_real_robot
        self.qHome = None
        self.start_robot()
        
    def start_robot(self):
	    # Create Configuration
        self.C = ry.Config()
        self.C.addFile(ry.raiPath('scenarios/pandaSingle.g'))
        self.C.addFrame('target')
        self.C.view(False)
        self.bot = ry.BotOp(self.C, useRealRobot=self.use_real_robot)

    def move_to_home(self, open_gripper=False, qHome=None):
	    # move to the home position specified by qHome
        if qHome is None:
            print("No home position given, abandoning move to home")
            return
        self.bot.moveTo(qHome, 0.2, True)
        while (self.bot.getTimeToEnd()>0.) :
            self.bot.sync(self.C,0)
        self.bot.sync(self.C, 0)
        self.qHome = qHome

    def IK(self, C, pos, use_quaternion=False):
		# set target to pos and quat
        if not use_quaternion:
            pos = self.PRYPoseToQuatPose(pos)
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

    def set_position(self, pos, use_quaternion=False, tau=None):
        q = self.IK(self.C, pos, use_quaternion=use_quaternion)
        if tau is None:
            self.bot.moveTo(q, 1., True)
        else:
            self.bot.move([q], [tau], False) #True
        self.bot.sync(self.C, 0)

    def get_position(self, use_quaternion=False):
        self.bot.sync(self.C, 0)
        #pos = self.arm.get_position()[1]
        position = self.C.getFrame('l_gripper').getPosition()
        quaternion = self.C.getFrame('l_gripper').getQuaternion()
        pos = [position[0], position[1], position[2], quaternion[0], quaternion[1], quaternion[2], quaternion[3]]
        if use_quaternion:
            return pos
        pos = self.quatPoseToPRYPose(pos)

        return np.array([pos[0], pos[1], pos[2], pos[3], pos[4], pos[5]]).astype(np.float32)
    
    def move_gripper(self, width= 0.07):
        self.bot.gripperMove(ry._left, width=width)

    def clear(self):
        del self.C
        del self.bot

    def PRYPoseToQuatPose(pos):
        #(yaw, pitch, roll) = (pos[3], pos[4], pos[5])
        (roll, pitch, yaw) = (pos[3], pos[4], pos[5])


        # convert pitch,roll and yaw to radians
        roll = roll * math.pi / 180
        pitch = pitch * math.pi / 180
        yaw = yaw * math.pi / 180


        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

        return [pos[0], pos[1], pos[2], qw, qx, qy, qz]
    

    def quatPoseToPRYPose(pos):
        quat = pos[3:]
        w = quat[0]
        x = quat[1]
        y = quat[2]
        z = quat[3]

        roll = math.atan2(2.0*(x*y + w*z), w*w + x*x - y*y - z*z)
        pitch = math.asin(-2.0*(x*z - w*y))
        yaw = math.atan2(2.0*(y*z + w*x), w*w - x*x - y*y + z*z)

        #convert pitch roll and yaw to angles
        roll = roll * 180 / math.pi
        pitch = pitch * 180 / math.pi
        yaw = yaw * 180 / math.pi

        pos = [pos[0], pos[1], pos[2], roll, pitch, yaw]

        return pos

    
        


address: str="tcp://localhost:69420"
verbose: int=1
FPS: int=30

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind(address)
input = np.array({"status": "listening"})
tau = 0.5
controller = FrankaController(use_real_robot=True)
if verbose:
    print("Listening for move commands...")
while True:
    
    # if verbose:
    #     print(f"Sending input {input} to be processed...")
    # to_send = pickle.dumps(input)
    # socket.send(to_send)
    message = socket.recv()
    message = pickle.loads(message)

    # convert from array to dictionary
    message = message[()]

    
    data = []
    # switch on message["command"]
    if message["command"] == "MOVE":
        print(f"Received MOVE command: {message['data']}")
        controller.set_position(message["data"], use_quaternion=True, tau=tau)
    elif message["command"] == "HOME":
        print(f"Received HOME command: {message['data']}")
        controller.move_to_home(qHome=message["data"])
    elif message["command"] == "GRIPPER":
        print(f"Received GRIPPER command: {message['data']}")
        controller.move_gripper(width=message["data"])
    elif message["command"] == "POSITION":
        print(f"Received POSITION command")
        data = controller.get_position(use_quaternion=True)
    elif message["command"] == "KILL":
        print("Received KILL command. Shutting down...")
        controller.clear()
        response = np.array({"status": "terminated", "data": [0]})
        to_send = pickle.dumps(response)
        socket.send(to_send)
        break

    # send response back  
    response = np.array({"status": "received", "data": data})
    to_send = pickle.dumps(response)
    socket.send(to_send)

    time_to_sleep = 1/FPS
    time.sleep(time_to_sleep)


