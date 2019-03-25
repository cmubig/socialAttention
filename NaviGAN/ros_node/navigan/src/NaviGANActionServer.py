#! /usr/bin/env python
import roslib
import rospy
import actionlib
from time import sleep

# add sgan path
import sys
sys.path.append('/home/nvidia/catkin_ws/src/navigan/src')

import math
import numpy as np
import tf
import torch

from cmu_perception_msgs.msg import TrackedObject, TrackedObjectSet
from arl_nav_msgs.msg import ExternalPathAction, ExternalPathFeedback, ExternalPathResult, ExternalPathGoal

from geometry_msgs.msg import Twist
from collections import deque, defaultdict
from threading import Lock

from sgan.models import TrajectoryGenerator, TrajectoryIntention
from sgan.utils import relative_to_abs

from attrdict import AttrDict

from matplotlib import pyplot as plt

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Pose
from std_msgs.msg import Header

def get_intention_generator(checkpoint, best=False):
    args = AttrDict(checkpoint['args'])
    generator = TrajectoryIntention(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        goal_dim=(2,),
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        batch_norm=args.batch_norm)
    if best:
        generator.load_state_dict(checkpoint['i_best_state'])
    else:
        generator.load_state_dict(checkpoint['i_state'])
    generator.cuda()
    generator.train()
    return generator

def get_force_generator(checkpoint, best=False):
    args = AttrDict(checkpoint['args'])
    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm)
    if best:
        generator.load_state_dict(checkpoint['g_best_state'])
    else:
        generator.load_state_dict(checkpoint['g_state'])
    generator.cuda()
    generator.train()
    return generator

class NaviGANsServer(object):
    def __init__(self, checkpoint):
        self.frame_rate = 0.4
        self.OBS_LEN = 8
        self.mutex = Lock()
        # Initializing variables for server
        self.active_peds_id = set()
        self.peds_pos = defaultdict(list)
        self.peds_pos_t = dict()
        self.robot_pos = []

        self.currentX = None
        self.currentY = None

        self.tracker_callback_last_call = None

        # Uncomment for visualization
        #ax = plt.gca()
        #ax.set_xlim([-10, 20])
        #ax.set_ylim([-10, 20])

        # Initializing service
        # TODO: define ExternalPathAction
        self.server = actionlib.SimpleActionServer('navigans_local_planner',
                ExternalPathAction, self.execute, False)
        # For debugging purpose, don't start the server
    #self.server.start()
        self.feedback = ExternalPathFeedback()

        # TODO: define ExternalPathFeedback
        # self.feedback = ExternalPathFeedback()

        self.tf_listener = tf.TransformListener()
        #self.husky_vel = rospy.Publisher('husky1/rcta_teleop/cmd_vel', 
        #                                 Twist, queue_size=1)

        self.controller_client = actionlib.SimpleActionClient('husky1/external_path', ExternalPathAction)
        print('Waiting for controller to connect...')
        self.controller_client.wait_for_server()

        while True:
            try:
                (trans,self.rot) = self.tf_listener.lookupTransform('/husky1/odom', 
                                                '/husky1/base', rospy.Time(0))
                break
            except (tf.LookupException, tf.ConnectivityException, \
                    tf.ExtrapolationException) as e:
                print e
                continue

        self.currentY = trans[0]
        self.currentX = trans[1]
        self.robot_pos = [np.array([self.currentX, self.currentY])]
        rospy.loginfo('[{}] Initialized...'.format(rospy.get_name()))

    def feedforward(self, obs_traj, obs_traj_rel, seq_start_end, goals_rel):
        """
        obs_traj: torch.Tensor([8, num_agents, 2])
        obs_traj_rel: torch.Tensor([8, num_agents, 2])
        seq_start_end: torch.Tensor([batch_size, 2])
        goals_rel: torch.Tensor([1, num_agents, 2])
        """
        rospy.loginfo('[{}] Feeding forward'.format(rospy.get_name()))

        with torch.no_grad():
            pred_traj_fake_rel = self.intention_generator(obs_traj, obs_traj_rel, seq_start_end, goal_input=goals_rel)
            pred_traj_fake_rel += self.force_generator(obs_traj, obs_traj_rel, seq_start_end)
            pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[0])
        return pred_traj_fake


    def execute(self, goal):
        goal_poses = goal.desired_path.poses
        N = len(goal_poses)
        for count, goal_pose in enumerate(goal_poses):
            ###################### GET THE GOAL OUT AND CONVERT TO NUMPY ######################
            goal_pos = goal_pose.pose.position
            goal = np.array([goal_pos.x, goal_pos.y])
            ##################################################################################

            next_waypoint = None
            #rospy.loginfo('[{}] Executing, NaviGAN planner, starting robot position: ({},{})'.format(rospy.get_name(), self.currentX, self.currentY))
            goal = torch.from_numpy(goal).type(torch.float)
            rate = rospy.Rate(1./self.frame_rate) # executing at 4hz
            while not rospy.is_shutdown():
                if self.server.is_preempt_requested():
                    # Client ask for preempt
                    rospy.loginfo('NaviGAN: Preempted')
                    self.server.set_preempted()
                    return

                if (self.currentX-goal[0])**2+(self.currentY-goal[1])**2 < self.safety_threshold**2:
                    # Move on to the next waypoint
                    rospy.loginfo('[{}] Goal reached!'.format(rospy.get_name()))
                    break
    
                # acquire mutex and collect information
                self.mutex.acquire()
                self.robot_pos.append(np.array([self.currentX, self.currentY]))
                if len(self.robot_pos) > self.OBS_LEN:
                    self.robot_pos = self.robot_pos[-self.OBS_LEN:]
    
                robot_pos = np.array(self.robot_pos[-self.OBS_LEN:])
                if robot_pos.ndim == 1:
                    robot_pos = robot_pos.reshape((1,2))
                #robot_pos = np.pad(robot_pos, ((self.OBS_LEN-len(robot_pos), 0), (0,0)), 'edge')

                peds_pos = []
                for ped_id in self.active_peds_id:
                    self.peds_pos[ped_id].append(np.array(self.peds_pos_t[ped_id]))
                    if len(self.peds_pos[ped_id]) > self.OBS_LEN:
                        self.peds_pos[ped_id] = self.peds_pos[ped_id][-self.OBS_LEN:]
                    print('ped {} pos: {}'.format(ped_id, self.peds_pos[ped_id]))
                    tmp = np.array(self.peds_pos[ped_id][-self.OBS_LEN:])
                    # prepend first position when traj too short
                    tmp = np.pad(tmp, ((self.OBS_LEN-len(tmp),0), (0,0)), 'edge')
                    peds_pos.append(tmp)
                    #plt.scatter(-tmp[-1,0], tmp[-1,1], c='k')
                    #plt.pause(0.0001)
                #plt.scatter(-robot_pos[-1,0], robot_pos[-1,1], c='b')
                #plt.pause(0.0001)
                #if next_waypoint is not None:
                #    plt.scatter(-next_waypoint[0], next_waypoint[1], c='r')
                #    plt.pause(0.0001)
                self.mutex.release()
    
                obs_traj = [robot_pos]+peds_pos
                obs_traj = np.array(obs_traj).transpose((1,0,2))
                
                obs_traj = torch.from_numpy(obs_traj).type(torch.float)
                obs_traj_rel = obs_traj - obs_traj[0,:,:]
    
                seq_start_end = torch.from_numpy(np.array([[0,obs_traj.shape[1]]]))
    
                goals_rel = goal-obs_traj[0,0,:]
                goals_rel = goals_rel.repeat(1,obs_traj.shape[1],1)
    
    
                # move everything to GPU
                obs_traj = obs_traj.cuda()
                obs_traj_rel = obs_traj_rel.cuda()
                seq_start_end = seq_start_end.cuda()
                goals_rel = goals_rel.cuda()
    
                pred_traj_fake = self.feedforward(obs_traj, obs_traj_rel, seq_start_end, goals_rel)
                print ('predicted trajectory: {}'.format(pred_traj_fake[:,0,:].cpu().numpy()))
    #next_waypoint = pred_traj_fake[self.OBS_LEN,0,:].cpu().numpy()
                ptf = pred_traj_fake[:,0,:].cpu().numpy()
#                for i in range(ptf.shape[0]):
#                    if (self.currentX-ptf[i,0])**2 + (self.currentY-ptf[i,1])**2 > (0.6)**2:
#                        break
                next_waypoint = ptf[i]
                rospy.loginfo('[{}] Next waypoint: {}; Current robot pos: ({},{})'.format(rospy.get_name(), next_waypoint, self.currentX, self.currentY))
    
        
                heading = tf.transformations.euler_from_quaternion(self.rot)[2]
                target_heading = math.atan2(next_waypoint[0]-self.currentX,
                                            next_waypoint[1]-self.currentY)

                ####################### TODO: PUBLISH TARGET HEADING AND NEXT WAYPOINT TO CONTROLER ###############

                message = ExternalPathGoal()

                message.desired_path.poses = []
                now = rospy.get_rostime()
                
                # Clean up previous goals
                self.controller_client.cancel_all_goals()

                for i, waypoint in enumerate(ptf):
                    if i > 0:
                        target_heading = math.atan2(ptf[i][0]-ptf[i-1][0],
                                                    ptf[i][1]-ptf[i-1][1])
                    aPose = Pose()
                    aPose.position.x = waypoint[0]
                    aPose.position.y = waypoint[1]
                    aPose.position.z = 0.
                    aPose.orientation.w = -1.
                    aPose.orientation.x = 0.
                    aPose.orientation.y = 0.
                    aPose.orientation.z = target_heading

                    aPoseStamped = PoseStamped()
                    aPoseStamped.pose = aPose

                    aPoseStamped.header.stamp = now + rospy.Duration.from_sec(self.frame_rate*(i+1))

                    message.desired_path.append(aPoseStamped)



                self.controller_client.send_goal(message)
                # Control the looping to specific rate
                rate.sleep()



        
            self.feedback.percent_complete = (count+1.)/N
            self.server.publish_feedback(self.feedback)

        # Done with all the goal poses, set succeeded
        self.server.set_succeeded()

    def trackerMsgCallback(self, tracked_object_set):
        # Control framerate
        if self.tracker_callback_last_call is not None and rospy.get_rostime() - self.tracker_callback_last_call < rospy.Duration.from_sec(self.frame_rate):
            return
        self.tracker_callback_last_call = rospy.get_rostime()

        while True:
            try:
                (trans,self.rot) = self.tf_listener.lookupTransform('/husky1/odom', 
                                                '/husky1/base', rospy.Time(0))
                break
            except (tf.LookupException, tf.ConnectivityException, \
                    tf.ExtrapolationException) as e:
                print e
                continue

        self.currentY = trans[0]
        self.currentX = trans[1]
#print('update robot position: ({}, {})'.format(self.currentX, self.currentY))
        #self.robot_pos.append(np.array([self.currentX, self.currentY]))
        #if len(self.robot_pos) > self.OBS_LEN:
        #    self.robot_pos = self.robot_pos[-self.OBS_LEN:]


        ####################### Mutex Block #######################
        self.mutex.acquire()
        self.active_peds_id = set()
        for tracked_object in tracked_object_set.objects:
            if not (tracked_object.classification == tracked_object.CLASS_Pedestrian): #or tracked_object.classification == tracked_object.CLASS_UnknownBig):
                continue
            print 'mover detected!'
            self.active_peds_id.add(tracked_object.object_id)
            self.peds_pos_t[tracked_object.object_id] = \
                    np.array([tracked_object.y, tracked_object.x])

        self.mutex.release()
        ####################### Mutex Block #######################



if __name__ == '__main__':

    CHECKPOINT = '/home/nvidia/catkin_ws/src/navigan/model/var_len_benchmark_zara1_with_model.pt'

    #rospy.init_node('navigans_control_server')
    rospy.init_node('navigans_local_planner')

    server = NaviGANsServer(torch.load(CHECKPOINT))
    print('[debug] subscribing to tracker...')
    rospy.Subscriber('/husky1_forward_lidar_tracking_data', TrackedObjectSet, server.trackerMsgCallback)

    #rospy.loginfo('[debug] navigating to fix goal...')
    #goal = np.array([-1., 16.])
    #sleep(2)
    #server.execute(goal)
    #server.simulate(goal)
    
    print('[debug] done subscribing, start spining')
    rospy.spin()

