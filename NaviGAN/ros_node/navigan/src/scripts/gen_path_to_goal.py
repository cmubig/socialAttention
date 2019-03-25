import argparse
import os
import torch
import imageio
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from attrdict import AttrDict

from sgan.data.loader import data_loader
from sgan.models import TrajectoryGenerator, TrajectoryDiscriminator, TrajectoryIntention
from sgan.losses import final_displacement_error
from sgan.utils import relative_to_abs, get_dset_path

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--dset_type', default='test', type=str)
parser.add_argument('--best', default=True, type=bool)
parser.add_argument('--plot_dir', default='./plots/trajs_to_goal/')
parser.add_argument('--sim_goal_itr', default=3, type=int)
parser.add_argument('--max_iter', default=5, type=int)
parser.add_argument('--tol', default=0.4, type=float)
parser.add_argument('--min_ped', default=7, type=int)

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

def get_discriminator(checkpoint, best=False):
    args = AttrDict(checkpoint['args'])
    discriminator = TrajectoryDiscriminator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        h_dim=args.encoder_h_dim_d,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_norm=args.batch_norm,
        d_type=args.d_type)
    if best:
        discriminator.load_state_dict(checkpoint['d_best_state'])
    else:
        discriminator.load_state_dict(checkpoint['d_state'])
    discriminator.cuda()
    discriminator.train()
    return discriminator

def gen_path_to_goal(
        args, loader, force_generator, intention_generator, discriminator, 
        tol, plot_dir=None, dset_type='val', max_iter=5, sim_goal_itr=3
    ):
    total_traj = 0
    count = 1
    guid = 0
    with torch.no_grad():
        for batch in loader:
            print('plotting trajectories for batch {}...'.format(count))
            count += 1
            batch = [tensor.cuda() for tensor in batch]

            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end, goals, goals_rel) = batch

            total_traj += pred_traj_gt.size(1)
            seq_size = pred_traj_gt.size(1)

            traj_gt = torch.cat([obs_traj, pred_traj_gt], dim=0)
            traj_gt_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
            pred_traj = torch.zeros(traj_gt.shape).cuda()
            pred_traj[:args.obs_len, :, :] = obs_traj

            obs_start = 0
            goals = pred_traj_gt[args.pred_len*sim_goal_itr-1, :, :].view(1, -1, 2)

            for itr in range(max_iter):
                for seq in range(1, seq_size-1):    # sacrify head and tail to avoid coner case
                    processed_obs_traj = torch.cat([traj_gt[obs_start:obs_start+args.obs_len, :seq,:],
                                            pred_traj[obs_start:obs_start+args.obs_len, seq, :].view(-1,1,2),
                                            traj_gt[obs_start:obs_start+args.obs_len, seq+1:, :]], dim=1)
                    processed_obs_traj_rel = processed_obs_traj - processed_obs_traj[0, :, :]
                    goals_rel = goals - processed_obs_traj[0, :, :]
                    pred_traj_fake_rel = intention_generator(
                        processed_obs_traj,
                        processed_obs_traj_rel,
                        seq_start_end, goal_input=goals_rel
                    )

                    pred_traj_fake_rel += force_generator(
                        processed_obs_traj,
                        processed_obs_traj_rel,
                        seq_start_end
                    )

                    pred_traj_fake = relative_to_abs(
                        pred_traj_fake_rel, processed_obs_traj[0, :, :]
                    )

                    pred_traj[(obs_start+args.obs_len):(obs_start+args.obs_len+args.pred_len), seq, :] = \
                        pred_traj_fake[:, seq, :]

                obs_start += args.pred_len
            
            # Record the trajectories
            # For each batch, draw only the first sample
            _plot_dir = plot_dir+args.dataset_name+'/'+dset_type+'/'+str(count)+'/'
            if not os.path.exists(_plot_dir):
                os.makedirs(_plot_dir)

            for seq in range(1, seq_size-1):
                fig = plt.figure()
                goal_point = goals[0, seq, :]
                whole_traj_fake = pred_traj[:, seq, :]
                for start_end in seq_start_end:
                    if start_end[1] > seq >= start_end[0]:
                        whole_traj_gt = traj_gt[:, start_end[0]:start_end[1], :]

                y_upper_limit = max([torch.max(whole_traj_fake[:, 1]).data, 
                                     torch.max(whole_traj_gt[:, :, 1]).data,
                                     goal_point[1].data]) + 1.
                y_lower_limit = min([torch.min(whole_traj_fake[:, 1]).data, 
                                     torch.min(whole_traj_gt[:, :, 1]).data,
                                     goal_point[1].data]) - 1.

                x_upper_limit = max([torch.max(whole_traj_fake[:, 0]).data, 
                                     torch.max(whole_traj_gt[:, :, 0]).data,
                                     goal_point[0].data]) + 1.
                x_lower_limit = min([torch.min(whole_traj_fake[:, 0]).data, 
                                     torch.min(whole_traj_gt[:, :, 0]).data,
                                     goal_point[0].data]) - 1.

                def plot_time_step(i):
                    fig, ax = plt.subplots()
                    ax.set_xlabel('time: {}'.format(i))
                    ax.plot(goal_point[0].cpu().numpy(), goal_point[1].cpu().numpy(), 'gx')
                    # plot last three point
                    gt_points_x = whole_traj_gt[max(i-2, 0):i+1,:,0].cpu().numpy().flatten()
                    gt_points_y = whole_traj_gt[max(i-2, 0):i+1,:,1].cpu().numpy().flatten()
                    ax.plot(gt_points_x, gt_points_y, 'b.')

                    fake_points_x = whole_traj_fake[max(i-2, 0):i+1,0].cpu().numpy()
                    fake_points_y = whole_traj_fake[max(i-2, 0):i+1,1].cpu().numpy()
                    if i >= args.obs_len:
                        if np.linalg.norm(whole_traj_fake[i, :].cpu().numpy() - goal_point.cpu().numpy()) < tol:
                            return None
                        ax.plot(fake_points_x, fake_points_y, 'r*')
                    else:
                        ax.plot(fake_points_x, fake_points_y, 'g.')

                    ax.set_ylim(y_lower_limit, y_upper_limit)
                    ax.set_xlim(x_lower_limit, x_upper_limit)

                    fig.canvas.draw()
                    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    plt.close(fig)

                    return image
                    
                img_seq = []
                for i in range(args.obs_len+args.pred_len*max_iter):
                    img = plot_time_step(i)
                    if img is None:
                        break
                    img_seq.append(img)

                imageio.mimsave(_plot_dir+str(guid)+'.gif', 
                                img_seq,
                                fps=2)
                guid += 1

        return


def main(args):
    if os.path.isdir(args.model_path):
        filenames = os.listdir(args.model_path)
        filenames.sort()
        paths = [
            os.path.join(args.model_path, file_) for file_ in filenames
        ]
    else:
        paths = [args.model_path]

    for path in paths:
        checkpoint = torch.load(path)
        force_generator = get_force_generator(checkpoint, best=args.best)
        intention_generator = get_intention_generator(checkpoint, best=args.best)
        discriminator = get_discriminator(checkpoint, best=args.best)
        _args = AttrDict(checkpoint['args'])
        tmp = _args.pred_len
        _args.pred_len *= args.max_iter
        path = get_dset_path(_args.dataset_name, args.dset_type)
        _, loader = data_loader(_args, path, shuffle=False, min_ped=args.min_ped)
        _args.pred_len = tmp
        gen_path_to_goal(    
            _args, loader, force_generator, intention_generator, discriminator, args.tol, 
            plot_dir=args.plot_dir, dset_type=args.dset_type, sim_goal_itr=args.sim_goal_itr, max_iter=args.max_iter)


if __name__ == '__main__':
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    main(args)
