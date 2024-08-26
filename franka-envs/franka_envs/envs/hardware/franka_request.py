import zmq
import pickle
import numpy as np


class FrankaRequestNode:
    def __init__(self, address: str="localhost"):
        self.address = "tcp://" + address + ":11111"
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(self.address)
        print("connected to ", "tcp://" + address + ":11111")

    def send_request(self,
            message: str,
            data: np.ndarray,
            verbose: int=1):

        input = np.array({"command": message, "data": data})

        if verbose:
            print(f"Sending input {input} to be processed...")
        to_send = pickle.dumps(input)
        self.socket.send(to_send)

        reply = self.socket.recv()
        reply = pickle.loads(reply)

        # tranform into dictionary
        reply = reply[()]
        if verbose:
            print(f"Received reply {reply} [ {input} ]")
            return reply

    def send_home_command(self, homePose=None):
        # This function sends the robot a command to slowlz go to the home position with the moveTo command
        # param homePose: The home position to move to in q space

        if homePose is None:
            print ("No home position given, abondaning move to home")
            return
        self.send_request("HOME", homePose)
        return

    def send_close_gripper_fully_command(self):
        # This function sends the robot a command to close the gripper fully
        self.send_request("GRIPPER", 0.008)
        return

    def send_open_gripper_fully_command(self):
        # This function sends the robot a command to open the gripper fully
        self.send_request("GRIPPER", 0.07)
        return

    def send_set_position_command(self, new_pos=None):
        # This function sends the robot a command to move to a new position
        # param new_pos: The new position to move to

        if new_pos is None:
            print ("No new position given, abondaning move to new position")
            return

        reply = self.send_request("MOVE", new_pos)
        if reply['status'] == 'torque limit exceeded':
            return False
        return True
    
    def send_get_position_command(self):
        # This function sends the robot a command to get the current position
        response = self.send_request("POSITION", 0, 1)
        return response["data"]




