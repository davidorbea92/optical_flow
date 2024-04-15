#!/usr/bin/env python3

import sys
import rospy
import moveit_commander
import moveit_msgs.msg
from geometry_msgs.msg import Pose
import numpy as np
from std_msgs.msg import Bool, String
from tf.transformations import quaternion_from_euler, euler_from_quaternion
import time
import math

arm_speed=0.6
class Movement:
    def __init__(self):
        time.sleep(10)
        self.robot_description = "wx250s/robot_description"
        self.robot = moveit_commander.RobotCommander(robot_description=self.robot_description, ns="wx250s")
        self.group = moveit_commander.MoveGroupCommander("interbotix_arm", robot_description=self.robot_description, ns="wx250s")
        self.gripper_group = moveit_commander.MoveGroupCommander("interbotix_gripper", robot_description=self.robot_description, ns="wx250s")

        self.invert = False  # This flag controls whether to add or subtract from the joint values

        """
        # Positions
        joint_goal[0] # waist
        joint_goal[1] # shoulder
        joint_goal[2] # elbow
        joint_goal[3] # forearm_roll
        joint_goal[4] # wrist_angle
        joint_goal[5] # wrist_rotate
        """

        self.home_pose = [0, 0, 0, 0, 0, 0] 
        self.sleep_pose = [0, -1.8, 1.55, 0, 0.8, 0]
        self.upright_pose = [0, 0, -1.5708, 0, 0, 0]
        self.closed_gripper = [0.015, -0.015]  # Joint values for closed gripper
        self.open_gripper = [0.037, -0.037]  # Joint values for open gripper
        self.adjustment_speed = 0.1  # Initial adjustment speed
        self.move_speed = 0.1  # Initial move speed

        self.init_pose=[0, -1.57, 1.2, 0, 0.31, 0] 
        self.inspection1=[0, -1.11, 0.15, 0, 0.9, 0] 
        self.inspection2=[-0.99, -0.45, 0.26, 1.43, 1.01, -1.32] 
        self.inspection3=[1.22, -0.06, -0.104, -1.5, 1.22,1.378] 
        self.inspection4=[1.01,-0.03,0.194,-1.885, 1.099, 2.21] 
        self.inspection5=[-0.873, -0.192,0.7854,1.99,1.01, -2.28]

        self.side1=[-34,14,-20,-91,62,90]
        self.side2=[31,14,-15,-91,-60,90]
        self.side21=[-90,-45,36,3,65,-5]
        self.side22=[-90,21,60,176,28,-179]
        self.side31=[90,-45,36,3,65,-5]
        self.side32=[90,23,60,176,27,-180]

        self.outdoors1=[0,0,-16,0,104,-2]
        self.outdoors2=[90,-18,13,1,66,-3]

        self.side1 = [i*math.pi/180 for i in self.side1]
        self.side2 = [i*math.pi/180 for i in self.side2]
        self.side21 = [i*math.pi/180 for i in self.side21]
        self.side22 = [i*math.pi/180 for i in self.side22]
        self.side31 = [i*math.pi/180 for i in self.side31]
        self.side32 = [i*math.pi/180 for i in self.side32]

        self.outdoors1 = [i*math.pi/180 for i in self.outdoors1]
        self.outdoors2= [i*math.pi/180 for i in self.outdoors2]

    def move_robot(self, key):
        if key == 'h':  # Go to home position
            self.move_to_position(self.home_pose)
        elif key == 's':  # Go to sleep position
            self.move_to_position(self.sleep_pose)
        elif key == 'd': # Go to upright position
            self.move_to_position(self.upright_pose)
        elif key == 'c':  # Close gripper
            self.move_gripper(self.closed_gripper)
        elif key == 'o':  # Open gripper
            self.move_gripper(self.open_gripper)
        elif key == 'q':  # Increase adjustment speed
            self.adjustment_speed += 0.05
            rospy.loginfo(f"Adjustment speed increased to {self.adjustment_speed}")
        elif key == 'z':  # Decrease adjustment speed
            self.adjustment_speed -= 0.05
            rospy.loginfo(f"Adjustment speed decreased to {self.adjustment_speed}")
        elif key == 'j':
            self.move_to_position(self.init_pose)
            self.move_to_position(self.inspection1)
            self.move_to_position(self.inspection2)
            self.move_to_position(self.inspection3)
            self.move_to_position(self.inspection4)
            self.move_to_position(self.inspection5)
            self.move_to_position(self.sleep_pose)
        elif key =='lu':
            self.move_to_position(self.init_pose)
            self.move_to_position(self.side2)
        elif key =='ru':
            self.move_to_position(self.init_pose)
            self.move_to_position(self.side1)
        elif key== 'rd':
            self.move_to_position(self.init_pose)
            self.move_to_position(self.side21)
            self.move_to_position(self.side22)
        elif key== 'ld':
            self.move_to_position(self.init_pose)
            self.move_to_position(self.side31)
            self.move_to_position(self.side32)
        elif key== 'od':
            self.move_to_position(self.init_pose)
            self.move_to_position(self.outdoors1)
            self.move_to_position(self.outdoors2)
        else:
            joint_goal = self.group.get_current_joint_values()
            if key == 't':  # Toggle to add mode
                self.invert = False
                rospy.loginfo("Mode switched to add")
            elif key == 'b':  # Toggle to subtract mode
                self.invert = True
                rospy.loginfo("Mode switched to subtract")
            else:
                index = ['u', 'i', 'o', 'j', 'k', 'l'].index(key)
                adjustment = self.adjustment_speed if not self.invert else -self.adjustment_speed
                joint_goal[index] += adjustment
                rospy.loginfo(f"Adjusted joint {key} by {adjustment}")

            # Apply the new joint positions
            self.group.set_max_velocity_scaling_factor(self.move_speed)
            self.group.go(joint_goal, wait=True)
            self.group.stop()  # Ensure there is no residual movement

    def move_to_position(self, position):
        self.group.clear_pose_targets()
        rospy.loginfo("Moving to the desired position!")
        self.group.set_max_velocity_scaling_factor(arm_speed)
        self.group.go(position, wait=True)
        self.group.stop()
        rospy.sleep(0.2)
    
    def move_gripper(self, joint_values):
        self.gripper_group.set_joint_value_target(joint_values)
        self.gripper_group.go(wait=True)
        self.gripper_group.stop()
        rospy.sleep(1)

    def callback(self, msg):
        self.move_robot(msg.data)

    def start(self):
        rospy.Subscriber('arm/position', String, self.callback)
        rospy.loginfo("Movement subscriber started")

if __name__ == '__main__':
    rospy.init_node('moving_robot', anonymous=True)
    move_robot = Movement()
    try:
        move_robot.start()
        rospy.spin()
    except rospy.ROSInterruptException:
        move_robot.group.stop()
        moveit_commander.roscpp_shutdown()



