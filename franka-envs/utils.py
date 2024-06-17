import math

def quatPoseToPRYPose(pos):
    quat = pos[3:]
    w = quat[0]
    x = quat[1]
    y = quat[2]
    z = quat[3]
    
    roll = math.atan2(2.0*(x*y + w*z), w*w + x*x - y*y - z*z)
    pitch = math.asin(-2.0*(x*z - w*y))
    yaw = math.atan2(2.0*(y*z + w*x), w*w - x*x - y*y + z*z)

    pos = [pos[0], pos[1], pos[2], roll, pitch, yaw]

    return pos

def PRYPoseToQuatPose(pos):
    (yaw, pitch, roll) = (pos[3], pos[4], pos[5])
    
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return [pos[0], pos[1], pos[2], qx, qy, qz, qw]