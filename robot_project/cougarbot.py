import roboticstoolbox as rtb
import math
from math import pi, sin, cos
import numpy as np
import time
import spatialmath as sm
from zmqRemoteApi import RemoteAPIClient
from matplotlib import pyplot as plt
from roboticstoolbox import *

"""
COUGARBOT
"""

# from CougarBot URDF
l1 = 0.50
l2 = 0.40
l3 = 0.40
l4 = 0.05

#function to build the matrix for each joint
def make_matrix(theta, alpha,a,d):
    matrix = np.array([[cos(theta), -sin(theta)*cos(alpha),  sin(theta)*sin(alpha), a*cos(theta)],
                       [sin(theta),  cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta)],
                       [         0,             sin(alpha),             cos(alpha),           d],
                       [         0,                      0,                      0,           1]])
    return matrix

def jacobian(q):
    theta1 = q[0]
    theta2 = q[1]
    theta3 = q[3]
    mtx = np.array([[-l2*sin(theta1+theta2)-l1*sin(theta1), -l2*sin(theta1+theta2), 0,  0],
                    [ l2*cos(theta1+theta2)+l1*cos(theta1),  l2*cos(theta1+theta2), 0,  0],
                    [                                    0,                      0, 0, -1],
                    [                                    0,                      0, 0,  0],
                    [                                    0,                      0, 0,  0],
                    [                                    1,                      1, 1,  0]])
    return mtx

# parametros DH

# revolução
a1 = 0.475
d1 = 0
# theta1 variável
alpha1 = 0

# revolução
a2 = 0.4
d2 = 0
# theta2 variável
alpha2 = pi

# prismática
a3 = 0
# variável 0 <= d3 <= 0.1
d3_lower_limit = 0
d3_upper_limit = 0.1
theta3 = 0
alpha3 = 0

# revolução
a4 = 0
d4 = 0
# theta4 variável
alpha4 = 0

# código abaixo monta o robô a partir dos parâmetros de DH
robot = DHRobot([
    RevoluteDH(d=l1, alpha=-pi/2),
    RevoluteDH(a=l2),
    RevoluteDH(a=l3),
    RevoluteDH(a=l4)
])

#robot.teach()

client = RemoteAPIClient()
sim = client.getObject('sim')

client.setStepping(True)
sim.startSimulation()

jointR1Handle = sim.getObject('/hip')
jointR2Handle = sim.getObject('/shoulder')
jointR3Handle = sim.getObject('/elbow')
jointR4Handle = sim.getObject('/wrist')
dummyHandle   = sim.getObject('/Dummy')

X_d1 = sim.getObjectPosition(dummyHandle,-1)
X_d2 = [np.pi, 0, sim.getObjectOrientation(dummyHandle,-1)[0]]
X_d = np.hstack([X_d1,X_d2])

#theta1, theta2, theta3 e d
q = [0,0,0,0]

q[0] = sim.getJointPosition(jointR1Handle)
q[1] = sim.getJointPosition(jointR2Handle)
q[2] = sim.getJointPosition(jointR3Handle)                      
q[3] = sim.getJointPosition(jointR4Handle)



X_c = robot.fkine(q)
X_c = X_c.A
X_c = X_c[:,3]
X_c = np.delete(X_c,3)
rpy = [np.pi, 0, sim.getObjectOrientation(jointR4Handle,-1)[0]]
X_c = np.hstack([X_c,rpy])

Ts = 0.03
e  = 1.60
X_e_list = []

pos_R1 = []
pos_R2 = []
pos_R3 = []
pos_R4 = []

while True:
    j = robot.jacobe(q)
    j_inv = np.linalg.pinv(j)

    X_e = X_d-X_c
    X_e_list.append(X_d-X_c)

    dq = j_inv@(X_e)
    q += dq*Ts

    sim.setJointPosition(jointR1Handle, q[0])
    time.sleep(0.05)
    sim.setJointPosition(jointR2Handle, q[1])
    time.sleep(0.05)
    sim.setJointPosition(jointR3Handle, q[2])
    time.sleep(0.05)
    sim.setJointPosition(jointR4Handle, q[3])
    time.sleep(0.05)

    pos_R1.append(q[0])
    pos_R2.append(q[1])
    pos_R3.append(q[2])
    pos_R4.append(q[3])

    X_c = robot.fkine(q)
    X_c = X_c.A
    X_c = X_c[:,3]
    X_c = np.delete(X_c,3)
    rpy = [np.pi,0,sim.getObjectOrientation(jointR4Handle,-1)[0]]
    X_c = np.hstack([X_c,rpy])

    X_d1 = sim.getObjectPosition(dummyHandle,-1)
    X_d2 = [np.pi,0,sim.getObjectOrientation(dummyHandle,-1)[0]]
    X_d = np.hstack([X_d1,X_d2])
    print(np.linalg.norm(X_e))
    if(np.linalg.norm(X_e) <= e):
        break

plt.title("Error by x, y, z, roll, pitch, and yaw")
plt.plot(X_e_list, label = ['x', 'y', 'z', 'roll', 'pitch', 'yaw'], linewidth=2)
plt.legend()
plt.show()

plt.title('Positions on Rotational Joint 1')
plt.plot(pos_R1, linewidth = 2)
plt.show()

plt.title('Positions on Rotational Joint 2')
plt.plot(pos_R2, linewidth = 2)
plt.show()

plt.title('Positions on Rotational Joint 3')
plt.plot(pos_R3, linewidth = 2)
plt.show()

plt.title('Positions on Rotational Joint 4')
plt.plot(pos_R4, linewidth = 2)
plt.show()

sim.stopSimulation()
