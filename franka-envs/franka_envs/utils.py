import math
import numpy as np

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

def quaternion_multiplication(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return [w, x, y, z]