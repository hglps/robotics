from math import pi, sin, cos
import numpy as np
import time
from zmqRemoteApi import RemoteAPIClient
from matplotlib import pyplot as plt
from roboticstoolbox import *

l1 =  0.475
l2 =  0.4

def jacobian(q):
    theta1 = q[0]
    theta2 = q[1]
    mtx = np.array([[-l2*sin(theta1+theta2)-l1*sin(theta1),-l2*sin(theta1+theta2) ,0,0],
                    [l2*cos(theta1+theta2)+l1*cos(theta1) ,l2*cos(theta1+theta2)  ,0,0],
                    [0                                    ,0                      ,0,-1],
                    [0                                    ,0                      ,0,0],
                    [0                                    ,0                      ,0,0],
                    [1                                    ,1                      ,1,0]])
    return mtx

# DH params

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
# d: 0 <= d3 <= 0.1
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
    RevoluteDH(a=a1, d=d1, alpha=alpha1),
    RevoluteDH(a=a2, d=d2, alpha=alpha2),
    RevoluteDH(a=a4, d=d4, alpha=alpha4),
    PrismaticDH(a=a3, theta=theta3, alpha=alpha3, qlim=[d3_lower_limit, d3_upper_limit])
])

client = RemoteAPIClient()
sim = client.getObject('sim')

client.setStepping(True)
sim.startSimulation()

jointR1Handle = sim.getObject('/MTB/axis')
jointR2Handle = sim.getObject('/MTB/link/axis')
jointR3Handle = sim.getObject('/MTB/link/axis/link/axis/axis')
jointP1Handle = sim.getObject('/MTB/link/axis/link/axis')
dummyHandle   = sim.getObject('/reference')

X_d1 = sim.getObjectPosition(dummyHandle,-1)
X_d2 = [np.pi,0,sim.getObjectOrientation(dummyHandle,-1)[0]]
X_d  = np.hstack([X_d1,X_d2])


#    theta1 , theta2, theta3, d
q = [0,       0,      0,      0] 

q[0] = sim.getJointPosition(jointR1Handle)
q[1] = sim.getJointPosition(jointR2Handle)
q[2] = sim.getJointPosition(jointR3Handle)
q[3] = sim.getJointPosition(jointP1Handle)

X_c = robot.fkine(q)
X_c = X_c.A # get Numpy repr
X_c = X_c[:,3]
X_c = np.delete(X_c,3)
rpy = [np.pi,0,sim.getObjectOrientation(jointP1Handle,-1)[0]] # get rpy from last joint
X_c = np.hstack([X_c,rpy])
print(X_c)

Ts = 0.04
e = 0.05
X_e_list = []

pos_R1 = []
pos_R2 = []
pos_R3 = []
pos_P1 = []

while True:
    j = jacobian(q)
    j_inv = np.linalg.pinv(j)

    X_e = X_d-X_c
    X_e_list.append(X_d-X_c)

    print("e:\n",X_e,"\n")
    dq = j_inv@(X_e)
    q += dq*Ts

    sim.setJointPosition(jointR1Handle, q[0])
    time.sleep(0.05)
    sim.setJointPosition(jointR2Handle, q[1])
    time.sleep(0.05)
    sim.setJointPosition(jointR3Handle, q[2])
    time.sleep(0.05)
    sim.setJointPosition(jointP1Handle, q[3])
    time.sleep(0.05)
    
    pos_R1.append(q[0])
    pos_R2.append(q[1])
    pos_R3.append(q[2])
    pos_P1.append(q[3])

    X_c = robot.fkine(q)
    X_c = X_c.A
    X_c = X_c[:,3]
    X_c = np.delete(X_c,3)
    rpy = [np.pi,0,sim.getObjectOrientation(jointP1Handle,-1)[0]]
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

plt.title('Positions on Prismatic Joint 1')
plt.plot(pos_P1, linewidth = 2)
plt.show()

sim.stopSimulation()
