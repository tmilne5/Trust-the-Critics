"""
Takes a list of critics and step sizes, and compute the average displacement of the generated
images for small perturbations about a single batch of noise. Ranks the input images
according to the size of these displacements, and plots them in that order.
"""

import os, sys

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'TTC_utils'))
import argparse
import time
import shutil
import random
import numpy as np
import torch
import pandas as pd
import pickle
from tqdm import tqdm
import matplotlib

matplotlib.use('Agg')

import dataloader
import networks
from generate_samples import generate_image
from generate_samples import save_individuals
from steptaker import steptaker

# get command line args~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

parser = argparse.ArgumentParser('TTC Evaluation Code')
parser.add_argument('--target', type=str, default='cifar10',
                    choices=['cifar10', 'mnist', 'fashion', 'celeba', 'monet', 'celebaHQ'])
parser.add_argument('--source', type=str, default='cifar10', choices=['noise', 'untrained_gen', 'photo'])
parser.add_argument('--temp_dir', type=str, required=True, help='directory where model state dicts are located')
parser.add_argument('--data', type=str, required=True, help='directory where data is located')
parser.add_argument('--model', type=str, default='dcgan',
                    choices=['dcgan', 'infogan', 'arConvNet', 'sndcgan', 'bsndcgan'])
parser.add_argument('--dim', type=int, default=64, help='int determining network dimensions')
parser.add_argument('--seed', type=int, default=-1, help='Set random seed for reproducibility')
parser.add_argument('--bs', type=int, default=128, help='batch size')
parser.add_argument('--num_workers', type=int, default=0, help='number of data loader processes')
parser.add_argument('--num_batch', type=int, default=100, help='number of times to redo generation')
parser.add_argument('--epsilon', type=float, default=0.01, help='std of noise for perturbations')

args = parser.parse_args()

temp_dir = args.temp_dir  # directory for temp saving

num_crit = len(os.listdir(os.path.join(temp_dir, 'model_dicts')))  # number of critics

# code to get deterministic behaviour
if args.seed != -1:  # if non-default seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # If true, optimizes convolution for hardware, but gives non-deterministic behaviour
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

# get dataloader
target_loader = getattr(dataloader, args.target)(args, train=False)

args.num_chan = target_loader.in_channels
args.hpix = target_loader.hpix
args.wpix = target_loader.wpix

source_loader = getattr(dataloader, args.source)(args, train=False)

if args.commonfake:
    gen = iter(source_loader)
    commonfake = next(gen)[0]

# begin definitions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

critic_list = [None] * num_crit

for i in range(num_crit):  # initialize pre-trained critics
    critic_list[i] = getattr(networks, args.model)(args.dim, args.num_chan, args.hpix, args.wpix)
    critic_list[i].load_state_dict(torch.load(os.path.join(temp_dir, 'model_dicts', 'critic{}.pth'.format(i))))

# Extract list of steps from log file
log = pd.read_pickle(os.path.join(temp_dir, 'log.pkl'))

steps_d = log['steps']
steps = []
for key in steps_d.keys():
    steps.append(steps_d[key])

print('Arguments:')
for p in vars(args).items():
    print('  ', p[0] + ': ', p[1])
print('\n')

use_cuda = torch.cuda.is_available()

if use_cuda:
    for i in range(num_crit):
        critic_list[i] = critic_list[i].cuda()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Sample source images and update them according to the critics and step sizes
gen = iter(source_loader)  # make dataloaders into iterables
tgen = iter(target_loader)

starttime = time.time()

# repeating seed selection again here to get same noise sequence
# code to get deterministic behaviour
if args.seed != -1:  # if non-default seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # If true, optimizes convolution for hardware, but gives non-deterministic behaviour
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

init_fake = next(gen)[0]
fake = init_fake.cuda()

finite_differences = []

for b_idx in tqdm(range(args.num_batch)):
    if b_idx > 0:
        noise = args.epsilon * torch.randn_like(init_fake)
        noise_mags = torch.norm(noise.flatten(1), dim=1) ** (-1)
        fake = (init_fake + noise).detach().clone()
        fake = fake.cuda()

    for i in range(num_crit):  # apply the steps of TTC
        eps = torch.tensor(steps[i]).cuda()
        fake = steptaker(fake, critic_list[i], eps, num_step=args.num_step)

    if b_idx == 0:
        base_point = fake.detach().clone()
    else:
        finite_differences.append(torch.norm(torch.flatten(base_point - fake, dims=1), dim=1) * noise_mags)

    generate_image(b_idx, fake.detach().cpu(), 'jpg', temp_dir)

finite_differences = torch.mean(torch.stack(finite_differences, dim=1), dim=1)
order = torch.argsort(finite_differences)
base_point = base_point[order, :, :, :]
generate_image('sorted', base_point.detach().cpu(), 'jpg', temp_dir)

