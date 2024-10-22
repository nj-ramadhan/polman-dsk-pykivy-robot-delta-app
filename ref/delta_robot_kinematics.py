#!/usr/bin/env python
"""
Delta Robot Kinematics code, liberally borrowed from tutorial:
http://forums.trossenrobotics.com/tutorials/introduction-129/delta-robot-kinematics-3276/
License: "You can freely use this code in your applications."

Note that the above tutorial defines the lengths f and e differently to me.  The
tutorial uses the side length of the equilateral triangle which has either the
servo output (for f) or the parallel link anchor (for e) in the middle of the side.

I use the convention that f or e is the displacement from the servo output/parallel
link anchor to the centre of the triangle, which is easier to measure.
"""

import math as maths
import matplotlib.pyplot as plt
import numpy as np

class DeltaPositionError(Exception):
    pass

class SimulatedDeltaBot(object):

    def __init__(self, servo_link_length, parallel_link_length, servo_displacement, effector_displacement):
        self.e = effector_displacement
        self.f = servo_displacement
        self.re = parallel_link_length
        self.rf = servo_link_length

    def forward(self, theta1, theta2, theta3):
        """ 
        Takes three servo angles in degrees.  Zero is horizontal.
        return (x,y,z) if point valid, None if not 
        """
        t = self.f-self.e

        theta1, theta2, theta3 = maths.radians(theta1), maths.radians(theta2), maths.radians(theta3)

        # Calculate position of leg1's joint.  x1 is implicitly zero - along the axis
        y1 = -(t + self.rf*maths.cos(theta1))
        z1 = -self.rf*maths.sin(theta1)

        # Calculate leg2's joint position
        y2 = (t + self.rf*maths.cos(theta2))*maths.sin(maths.pi/6)
        x2 = y2*maths.tan(maths.pi/3)
        z2 = -self.rf*maths.sin(theta2)

        # Calculate leg3's joint position
        y3 = (t + self.rf*maths.cos(theta3))*maths.sin(maths.pi/6)
        x3 = -y3*maths.tan(maths.pi/3)
        z3 = -self.rf*maths.sin(theta3)

        # From the three positions in space, determine if there is a valid
        # location for the effector
        dnm = (y2-y1)*x3-(y3-y1)*x2
    
        w1 = y1*y1 + z1*z1
        w2 = x2*x2 + y2*y2 + z2*z2
        w3 = x3*x3 + y3*y3 + z3*z3

        # x = (a1*z + b1)/dnm
        a1 = (z2-z1)*(y3-y1)-(z3-z1)*(y2-y1)
        b1 = -((w2-w1)*(y3-y1)-(w3-w1)*(y2-y1))/2.0

        # y = (a2*z + b2)/dnm;
        a2 = -(z2-z1)*x3+(z3-z1)*x2
        b2 = ((w2-w1)*x3 - (w3-w1)*x2)/2.0

        # a*z^2 + b*z + c = 0
        a = a1*a1 + a2*a2 + dnm*dnm
        b = 2*(a1*b1 + a2*(b2-y1*dnm) - z1*dnm*dnm)
        c = (b2-y1*dnm)*(b2-y1*dnm) + b1*b1 + dnm*dnm*(z1*z1 - self.re*self.re)
 
        # discriminant
        d = b*b - 4.0*a*c
        if d < 0:
            return None # non-existing point

        z0 = -0.5*(b+maths.sqrt(d))/a
        x0 = (a1*z0 + b1)/dnm
        y0 = (a2*z0 + b2)/dnm
        return (x0,y0,z0)


    def _calcAngleYZ(self, x0, y0, z0):
        y1 = -self.f
        y0 -= self.e
        a = (x0*x0 + y0*y0 + z0*z0 + self.rf*self.rf - self.re*self.re - y1*y1)/(2*z0)
        b = (y1-y0)/z0
        d = -(a + b*y1)*(a + b*y1) + self.rf*(b*b*self.rf + self.rf)
        if d < 0:
            raise DeltaPositionError()
        yj = (y1 - a*b - maths.sqrt(d))/(b*b + 1)
        zj = a + b*yj
        theta = 180.0*maths.atan(-zj/(y1-yj))/maths.pi
        if yj>y1:
            theta += 180.0
        return theta


    def inverse(self, x0, y0, z0):
        """
        Takes position and returns three servo angles, or 0,0,0 if not possible
        return (x,y,z) if point valid, None if not
        """
        cos120 = maths.cos(2.0*maths.pi/3.0)
        sin120 = maths.sin(2.0*maths.pi/3.0)

        try:
            theta1 = self._calcAngleYZ(x0, y0, z0)
            theta2 = self._calcAngleYZ(x0*cos120 + y0*sin120, y0*cos120 - x0*sin120, z0) # rotate +120 deg
            theta3 = self._calcAngleYZ(x0*cos120 - y0*sin120, y0*cos120 + x0*sin120, z0) # rotate -120 deg

            return theta1, theta2, theta3
        except DeltaPositionError:
            print("error")
            return 0,0,0


if __name__ == '__main__':
    # ## versi TA
    # servo_link_length = 295.0
    # parallel_link_length = 495.0
    # servo_displacement = 230.0
    # effector_displacement = 80.0

    tool_offset = 20.0

    # versi Penelitian POLeLAND
    servo_link_length = 640.0
    parallel_link_length = 840.0
    servo_displacement = 225.7
    effector_displacement = 60.0

    bot = SimulatedDeltaBot(servo_link_length, parallel_link_length,
                            servo_displacement, effector_displacement)
    

    # robot modelling
    # calculate IK for control system, input destined coordinate, output joint angle 
    ik_result = bot.inverse(0.0 ,500.0, -500.0)
    print(ik_result)

    servo_angle = np.array([ik_result[0],ik_result[1],ik_result[2]], dtype=int)

    cos120 = maths.cos(2.0*maths.pi/3.0)
    sin120 = maths.sin(2.0*maths.pi/3.0)

    fk_result = bot.forward(*servo_angle)
    print(fk_result)

    base = np.array([[0, -servo_displacement, 0],
            [sin120*servo_displacement,-cos120*servo_displacement,0],
            [-sin120*servo_displacement,-cos120*servo_displacement,0]])
    
    platform = np.array([[fk_result[0], fk_result[1]-effector_displacement,fk_result[2]],
            [fk_result[0]+sin120*effector_displacement,fk_result[1]-cos120*effector_displacement,fk_result[2]],
            [fk_result[0]-sin120*effector_displacement,fk_result[1]-cos120*effector_displacement,fk_result[2]]])
    
    t = servo_displacement-effector_displacement
    theta1, theta2, theta3 = maths.radians(servo_angle[0]), maths.radians(servo_angle[1]), maths.radians(servo_angle[2])
    # Calculate position of leg1's joint.  x1 is implicitly zero - along the axis
    y1 = -(t + servo_link_length*maths.cos(theta1))
    z1 = -servo_link_length*maths.sin(theta1)
    # Calculate leg2's joint position
    y2 = (t + servo_link_length*maths.cos(theta2))*maths.sin(maths.pi/6)
    x2 = y2*maths.tan(maths.pi/3)
    z2 = -servo_link_length*maths.sin(theta2)
    # Calculate leg3's joint position
    y3 = (t + servo_link_length*maths.cos(theta3))*maths.sin(maths.pi/6)
    x3 = -y3*maths.tan(maths.pi/3)
    z3 = -servo_link_length*maths.sin(theta3)

    joint = np.array([[0,y1,z1],
            [x2,y2,z2],
            [x3,y3,z3]])
    
   
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection='3d')
    ax.scatter(xs=[x for x,y,z in base] ,ys=[y for x,y,z in base],zs=[z for x,y,z in base])
    ax.scatter(xs=[x for x,y,z in platform] ,ys=[y for x,y,z in platform],zs=[z for x,y,z in platform])
    
    for i in range(3):
        ax.plot([base.T[0,i] ,joint.T[0,i]],[base.T[1,i],joint.T[1,i]],[base.T[2,i],joint.T[2,i]])
    for i in range(3):
        ax.plot([joint.T[0,i] ,platform.T[0,i]],[joint.T[1,i],platform.T[1,i]],[joint.T[2,i],platform.T[2,i]])
    plt.show()


    # # calculate FK for workspace simulation, input joint angle, output end-effector coordinate 
    # step = 5
    # minServo = -30
    # maxServo = 70

    # points = []
    # for t1 in range(minServo, maxServo, step):
    #     for t2 in range(minServo, maxServo, step):
    #         for t3 in range(minServo, maxServo, step):
    #             servos = (t1, t2, t3)
    #             points.append(bot.forward(*servos))
    #             there_and_back = bot.inverse(*bot.forward(*servos))
    #             err = map(lambda a,b: abs(a-b), servos, there_and_back)
    #             if max(err) > 0.0000000000001:
    #                 print(servos, there_and_back, err)

    # fig = plt.figure()
    # ax = fig.add_subplot(1,1,1, projection='3d')
    # surf = ax.scatter(xs=[x for x,y,z in points] ,ys=[y for x,y,z in points],zs=[z for x,y,z in points])

    # plt.show()

    # # # simulate trajectory when motor speed is constant
    # base = np.array([[0, -servo_displacement, 0],
    #         [sin120*servo_displacement,-cos120*servo_displacement,0],
    #         [-sin120*servo_displacement,-cos120*servo_displacement,0]])
    
    # platform = np.array([[fk_result[0], fk_result[1]-effector_displacement,fk_result[2]],
    #         [fk_result[0]+sin120*effector_displacement,fk_result[1]-cos120*effector_displacement,fk_result[2]],
    #         [fk_result[0]-sin120*effector_displacement,fk_result[1]-cos120*effector_displacement,fk_result[2]]])
    
    # t = servo_displacement-effector_displacement
    # theta1, theta2, theta3 = maths.radians(servo_angle[0]), maths.radians(servo_angle[1]), maths.radians(servo_angle[2])
    # # Calculate position of leg1's joint.  x1 is implicitly zero - along the axis
    # y1 = -(t + servo_link_length*maths.cos(theta1))
    # z1 = -servo_link_length*maths.sin(theta1)
    # # Calculate leg2's joint position
    # y2 = (t + servo_link_length*maths.cos(theta2))*maths.sin(maths.pi/6)
    # x2 = y2*maths.tan(maths.pi/3)
    # z2 = -servo_link_length*maths.sin(theta2)
    # # Calculate leg3's joint position
    # y3 = (t + servo_link_length*maths.cos(theta3))*maths.sin(maths.pi/6)
    # x3 = -y3*maths.tan(maths.pi/3)
    # z3 = -servo_link_length*maths.sin(theta3)

    # joint = np.array([[0,y1,z1],
    #         [x2,y2,z2],
    #         [x3,y3,z3]])
    
   
    # fig = plt.figure()
    # ax = fig.add_subplot(1,1,1, projection='3d')
    # ax.scatter(xs=[x for x,y,z in base] ,ys=[y for x,y,z in base],zs=[z for x,y,z in base])
    # ax.scatter(xs=[x for x,y,z in platform] ,ys=[y for x,y,z in platform],zs=[z for x,y,z in platform])
    
    # for i in range(3):
    #     ax.plot([base.T[0,i] ,joint.T[0,i]],[base.T[1,i],joint.T[1,i]],[base.T[2,i],joint.T[2,i]])
    # for i in range(3):
    #     ax.plot([joint.T[0,i] ,platform.T[0,i]],[joint.T[1,i],platform.T[1,i]],[joint.T[2,i],platform.T[2,i]])
    # # calculate IK for control system, input destined coordinate, output joint angle 
    # ik_result_begin = bot.inverse(0.0 ,0.0, -216.0)
    # print("Sudut Joint awal",ik_result_begin)
    # servo_angle_begin = np.array([ik_result_begin[0],ik_result_begin[1],ik_result_begin[2]], dtype=int)
    # # calculate IK for control system, input destined coordinate, output joint angle 
    # ik_result_end = bot.inverse(0.0 ,700.0, -1200.0)
    # print("Sudut Joint akhir",ik_result_end)
    # servo_angle_end = np.array([ik_result_end[0],ik_result_end[1],ik_result_end[2]], dtype=int)
    # # calculate joint velocity
    # pps = 180 # pulse per second
    # ppr = 400 # pulse per rotation
    # motor_rpm = (pps / ppr) * 60 # rotation per second = pulse per second / pulse per rotation, rpm = rotation per second * 60 second

    # joint_vel = motor_rpm * 360 / 60 / 25 # Angle per second = motor_rpm * 360 / 60 / gear_ratio
    # step_interval = 1 / joint_vel
    # print("Kecepatan joint",joint_vel,"[°/s], Delay waktu per titik trajectory", step_interval,"[s/step]")
    # # initialize trajectory points, and servo angle
    # traj = []
    # traj_vel = []
    # servo_diff = servo_angle_end - servo_angle_begin
    # servos = servo_angle_begin
    # for i in range(np.max(servo_diff)):
    #     servos = list(servos)
    #     if servos[0] < servo_angle_end[0]:
    #         servos[0] = servo_angle_begin[0] + i
    #     if servos[1] < servo_angle_end[1]:
    #         servos[1] = servo_angle_begin[1] + i
    #     if servos[2] < servo_angle_end[2]:
    #         servos[2] = servo_angle_begin[2] + i
    #     servos = tuple([servos[0], servos[1], servos[2]])
    #     print("Step ke" ,i ,", Sudut Joint" ,servos)

    #     traj.append(bot.forward(*servos))

    #     there_and_back = bot.inverse(*bot.forward(*servos))
    #     err = map(lambda a,b: abs(a-b), servos, there_and_back)
    #     if max(err) > 0.0000000000001:
    #         print(servos, there_and_back, err)
    
    # traj = np.round(np.array(traj), decimals=2)
    # print("Koordinat end effector" ,traj)
    # np.savetxt("trajectory.csv",traj,delimiter =";", fmt='%.2f')
    
    # # fig = plt.figure()
    # # ax = fig.add_subplot(1,1,1, projection='3d')
    # surf = ax.scatter(xs=[x for x,y,z in traj] ,ys=[y for x,y,z in traj],zs=[z for x,y,z in traj])
    # plt.show()
    
    

