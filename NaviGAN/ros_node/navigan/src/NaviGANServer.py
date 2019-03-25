#! /usr/bin/env python
import roslib
import rospy
import actionlib
from time import sleep

# add sgan path
import sys
#sys.path.append('/home/nvidia/catkin_ws/src/navigan/src')

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
        self.OBS_LEN = 8
        self.mutex = Lock()
        # Initializing variables for server
        self.active_peds_id = set()
        self.peds_pos = defaultdict(list)
        # self.robot_pos = [(0,0)]
        self.robot_pos = []
        # self.currentX = 0
        # self.currentY = 0
        self.currentX = None
        self.currentY = None
        self.rot = 0
        self.safety_threshold = .7 # don't run into ped
        self.intention_generator = get_intention_generator(checkpoint)
        self.force_generator = get_force_generator(checkpoint)
        
        self.initial_threshold = .1
        self.turn_threshold = .001
        self.angular_displacement_threshold = self.initial_threshold
        self.cmd = Twist()

        self.cmd.linear.y = 0
        self.cmd.linear.z = 0
        self.cmd.angular.x = 0
        self.cmd.angular.y = 0
        self.cmd.angular.z = 0

        self.f_vel = 0.3
        self.a_vel = 0.9

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
        self.husky_vel = rospy.Publisher('husky1/rcta_teleop/cmd_vel', 
                                         Twist, queue_size=1)
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
    
    def simulate(self, goal):
        ax = plt.gca()
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])

        plt.plot(goal[0], goal[1], c='r', marker='x')
        plt.pause(0.0001)
        rate = rospy.Rate(4) # executing at 4hz
        first_flag = True
        goal = torch.from_numpy(goal).type(torch.float)
        while True:
            self.mutex.acquire()
            if first_flag:
                robot_pos = np.array(self.robot_pos[-self.OBS_LEN:])
                robot_pos = np.pad(robot_pos, ((self.OBS_LEN-len(robot_pos), 0), (0,0)), 'edge')
                plt.plot(robot_pos[-1][0], robot_pos[-1][1], c='k', marker='o')
                plt.pause(0.0001)
                first_flag = False
            else:
                robot_pos = np.concatenate([robot_pos[1:, :], next_waypoint.reshape(1,2)], axis=0)

            peds_pos = []
            for ped_id in self.active_peds_id:
                print('ped {} pos: {}'.format(ped_id, self.peds_pos[ped_id][-1]))
                tmp = np.array(self.peds_pos[ped_id][-self.OBS_LEN:])
                # prepend first position when traj too short
                tmp = np.pad(tmp, ((self.OBS_LEN-len(tmp),0), (0,0)), 'edge')
                peds_pos.append(tmp)
            
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
            next_waypoint = pred_traj_fake[0,0,:].cpu().numpy()
            future = pred_traj_fake[:,0,:].cpu().numpy()
            plt.scatter(future[:,0], future[:,1], c='r', s=2, marker='o')
            plt.plot(next_waypoint[0], next_waypoint[1], c='b', marker='o')
            plt.pause(0.0001)
            
            raw_input('press any key to continue....')
            
            if np.sqrt((next_waypoint[0]-goal[0])**2 + (next_waypoint[1]-goal[1])**2) < 0.5:
                print('Done!')
                break

            rate.sleep()


    def execute(self, goal):
        rospy.loginfo('[{}] Executing, NaviGAN planner, starting robot position: ({},{})'.format(rospy.get_name(), self.currentX, self.currentY))
        goal = torch.from_numpy(goal).type(torch.float)
        while not rospy.is_shutdown():
            if (self.currentX-goal[0])**2+(self.currentY-goal[1])**2 < self.safety_threshold**2:
                rospy.loginfo('[{}] Goal reached!'.format(rospy.get_name()))
                break

            if len(self.active_peds_id) == 0:
                rospy.loginfo('[{}] No peds detected yet, sleep for 0.4 sec'.format(rospy.get_name()))
#                sleep(0.4)
#                continue
            # acquire mutex and collect information
            self.mutex.acquire()
            robot_pos = np.array(self.robot_pos[-self.OBS_LEN:])
            robot_pos = np.pad(robot_pos, ((self.OBS_LEN-len(robot_pos), 0), (0,0)), 'edge')
            peds_pos = []
            for ped_id in self.active_peds_id:
                print('ped {} pos: {}'.format(ped_id, self.peds_pos[ped_id][-1]))
                tmp = np.array(self.peds_pos[ped_id][-self.OBS_LEN:])
                # prepend first position when traj too short
                tmp = np.pad(tmp, ((self.OBS_LEN-len(tmp),0), (0,0)), 'edge')
                peds_pos.append(tmp)
            
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
            next_waypoint = pred_traj_fake[0,0,:].cpu().numpy()
            rospy.loginfo('[{}] Next waypoint: {}; Current robot pos: ({},{})'.format(rospy.get_name(), next_waypoint, self.currentX, self.currentY))

####################################### VIRTUAL TEST DRIVE ###################################

            self.currentX = next_waypoint[0]
            self.currentY = next_waypoint[1]
            self.robot_pos.append(next_waypoint)
            sleep(0.1)
            continue
##############################################################################################
    
            heading = tf.transformations.euler_from_quaternion(self.rot)[2]
            target_heading = math.atan2(next_waypoint[0]-self.currentX,
                                        next_waypoint[1]-self.currentY)
    
            angle_mag = abs(heading - target_heading)
            if angle_mag > self.angular_displacement_threshold: # turn
                raw_z = self.a_vel*abs(heading-target_heading)**2
                self.cmd.angular.z = -raw_z if heading > target_heading else raw_z
                self.angular_displacement_threshold = self.turn_threshold
            else: # no turn
                self.cmd.angular.z = 0
                self.angular_displacement_threshold = self.initial_threshold
    
            self.cmd.linear.x = self.f_vel
            # send out cmd
            # self.cmd.linear.x = 0.1
            self.husky_vel.publish(self.cmd)
            sleep(0.3)

        self.feedback.percent_complete = 0.0101
        self.server.publish_feedback(self.feedback)
        self.server.set_succeeded()

    def execute_ped_follower(self, goal):
        rospy.loginfo('[{}] Executing, Ped Follower'.format(rospy.get_name()))
        while not rospy.is_shutdown():
            ####################### Mutex Block #######################
            self.mutex.acquire()
            # TODO: look at the peds_pos and generate next waypoint
            if len(self.active_peds_id) == 0:
                self.mutex.release()
                continue
            all_pos = [self.peds_pos[ped_id][-1] for ped_id in self.active_peds_id]
            all_pos = np.array(all_pos)
            robot_pos = np.array(self.robot_pos[-1])
            robot_pos = robot_pos.reshape([1,2])
            p_dist = (all_pos-robot_pos)**2
            p_dist = np.sum(p_dist, axis=1)
    
            next_waypoint = all_pos[p_dist.argmin(), :]
            self.mutex.release()
            ####################### Mutex Block #######################
            if (self.currentX-next_waypoint[0])**2+\
               (self.currentY-next_waypoint[1])**2 < self.safety_threshold**2:

                self.cmd.angular.z = 0
                self.cmd.linear.z = 0
                self.husky_vel.publish(self.cmd)
                rospy.loginfo('[{}] Stop for safety reason'.format(rospy.get_name()))
                return
    
            heading = tf.transformations.euler_from_quaternion(self.rot)[2]
            target_heading = math.atan2(next_waypoint[0]-self.currentX,
                                        next_waypoint[1]-self.currentY)
    
            angle_mag = abs(heading - target_heading)
            if angle_mag > self.angular_displacement_threshold: # turn
                raw_z = self.a_vel*abs(heading-target_heading)**2
                self.cmd.angular.z = -raw_z if heading > target_heading else raw_z
                self.angular_displacement_threshold = self.turn_threshold
            else: # no turn
                self.cmd.angular.z = 0
                self.angular_displacement_threshold = self.initial_threshold
    
            self.cmd.linear.x = self.f_vel
            # send out cmd
            self.husky_vel.publish(self.cmd)

        self.feedback.percent_complete = 0.0101
        self.server.publish_feedback(self.feedback)
        self.server.set_succeeded()


    def trackerMsgCallback(self, tracked_object_set):
        if self.currentY != None:
            return
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
        self.robot_pos.append(np.array([self.currentX, self.currentY]))
        if len(self.robot_pos) > self.OBS_LEN:
            self.robot_pos = self.robot_pos[-self.OBS_LEN:]


        ####################### Mutex Block #######################
        self.mutex.acquire()
        self.active_peds_id = set()
        for tracked_object in tracked_object_set.objects:
            if not (tracked_object.classification == tracked_object.CLASS_Pedestrian): #or tracked_object.classification == tracked_object.CLASS_UnknownBig):
                continue
#            print 'mover detected!'
            self.active_peds_id.add(tracked_object.object_id)
            self.peds_pos[tracked_object.object_id].append(
                    np.array([tracked_object.x, tracked_object.y]))
            if len(self.peds_pos[tracked_object.object_id]) > self.OBS_LEN:
                self.peds_pos[tracked_object.object_id] = self.peds_pos[tracked_object.object_id][-self.OBS_LEN:]

        self.mutex.release()
        ####################### Mutex Block #######################



if __name__ == '__main__':

    print 'Finishing imports...'
    CHECKPOINT = '/home/nvidia/catkin_ws/src/navigan/model/benchmark_zara1_with_model.pt'
    #CHECKPOINT = rospy.get_param('~modelName')
    
    #rospy.init_node('navigans_control_server')
    rospy.init_node('navigans_local_planner')

    server = NaviGANsServer(torch.load(CHECKPOINT))
    print('[debug] subscribing to tracker...')
#    tracker_topic = rospy.get_param('~subscriberTopic')

    rospy.Subscriber('/husky1_forward_lidar_tracking_data', TrackedObjectSet, server.trackerMsgCallback)

    rospy.loginfo('[debug] navigating to fix goal...')
    goal = np.array([0., 5.])
    sleep(2)
    #server.execute(goal)
    server.simulate(goal)
    
    #print('[debug] done subscribing, start spining')
    #rospy.spin()
