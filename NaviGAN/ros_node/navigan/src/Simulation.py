from sgan.models import TrajectoryGenerator, TrajectoryIntention
from sgan.utils import relative_to_abs

from attrdict import AttrDict

import numpy as np
import torch

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

CHECKPOINT = '/home/nvidia/catkin_ws/src/navigan/model/var_len_benchmark_zara1_with_model.pt'
CHECKPOINT = torch.load(CHECKPOINT)
intention_generator = get_intention_generator(CHECKPOINT)
force_generator = get_force_generator(CHECKPOINT)

color = ['r', 'b', 'k', 'y', 'g', 'o']


def feedforward(obs_traj, obs_traj_rel, seq_start_end, goals_rel):
    """
    obs_traj: torch.Tensor([8, num_agents, 2])
    obs_traj_rel: torch.Tensor([8, num_agents, 2])
    seq_start_end: torch.Tensor([batch_size, 2])
    goals_rel: torch.Tensor([1, num_agents, 2])
    """
    with torch.no_grad():
        pred_traj_fake_rel = intention_generator(obs_traj, obs_traj_rel, seq_start_end, goal_input=goals_rel)
        pred_traj_fake_rel += force_generator(obs_traj, obs_traj_rel, seq_start_end)
        pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[0])
    return pred_traj_fake


def self_play(start, goal):
    """
    start shape: [num_peds, 2]
    goal shape: [num_peds, 2]
    """
    ax = plt.gca()
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    
    num_ped = len(start)
    
    for i in range(num_ped):
        plt.scatter(start[i,0], start[i,1], c=color[i])
        plt.scatter(goal[i,0], goal[i,1], c=color[i], marker='x')
    plt.pause(0.0001)

    
    goal = torch.from_numpy(goal.reshape(1, len(goal), 2)).type(torch.float)
    first_flag = True
    while True:
        if first_flag:
            obs_traj_np = np.pad(start.reshape(1,len(start),2), ((7,0), (0, 0), (0,0)), 'edge')
            first_flag = False
        else:
            obs_traj_np = np.concatenate([obs_traj_np[1:, :, :], next_waypoint.reshape(1,-1,2)], axis=0)

        obs_traj = torch.from_numpy(obs_traj_np).type(torch.float)
        obs_traj_rel = obs_traj - obs_traj[0,:,:]

        seq_start_end = torch.from_numpy(np.array([[0, obs_traj.shape[1]]]))
        goals_rel = goal-obs_traj[0,:,:]

        obs_traj = obs_traj.cuda()
        obs_traj_rel = obs_traj_rel.cuda()
        seq_start_end = seq_start_end.cuda()
        goals_rel = goals_rel.cuda()

        pred_traj_fake = feedforward(obs_traj, obs_traj_rel, seq_start_end, goals_rel)

        next_waypoint = pred_traj_fake[0,:,:].cpu().numpy()
        for i in range(num_ped):
            plt.scatter(next_waypoint[i,0], next_waypoint[i,1], c=color[i])
        plt.pause(0.0001)
        raw_input('Press ENTER to continue')

        


if __name__ == '__main__':
    start = np.array([np.array([-3.,0.]),
                      np.array([0.,3.])])
    goal = np.array([np.array([3.,0.]),
                     np.array([0.,-3.])])
    self_play(start, goal)
