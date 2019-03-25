#! /usr/bin/env python
import roslib
import rospy
import actionlib

from arl_nav_msgs.msg import ExternalPathAction, ExternalPathGoal

if __name__ == '__main__':
    print('[debug] initializing node...')
    rospy.init_node('navigans_control_client')
    print('[debug] done initializing node, initializing client...')
    client = actionlib.SimpleActionClient('navigans_local_planner', 
                                          ExternalPathAction)
    print('[debug] done initializing client, connecting to server...')
    client.wait_for_server()
    print('[debug] server connected, sending request...')
    goal = ExternalPathGoal()

    
    client.send_goal(goal)
    while not rospy.is_shutdown():
        if not client.wait_for_result(rospy.Duration.from_sec(1000.)):
            client.cancel_all_goals()

    client.cancel_all_goals()
