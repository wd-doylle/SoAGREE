import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from soagree import SoAGREE
from dataset import GATDataset
from predict import predict


def validate(model,val_loader,loss,cuda):
    t = time.time()
    
    model.eval()
    losses_val = []
    for batch_id, (repo,users,user_neighbors,teams,team_users,target_users, target_team) in enumerate(val_loader):
        if cuda:
            try:
                repo = repo.cuda()
                users = users.cuda()
                user_edges = user_edges.cuda()
                teams = teams.cuda()
                target = target.cuda()
            except:
                continue
        output = model(repo, teams, team_users, users, user_neighbors)
        loss_val = loss(output,target_team)
        losses_val.append(loss_val.item())

    avg_loss_val = torch.mean(torch.tensor(losses_val))

    print('loss_val: {:.8f}'.format(avg_loss_val),
          'time: {:.4f}s'.format(time.time() - t))

    return avg_loss_val


if __name__ == '__main__':

    t_total = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--repo_feature_file', type=str, default="repo_embedding.json")
    parser.add_argument('--team_feature_file', type=str, default="team_embedding.json")
    parser.add_argument('--user_feature_file', type=str, default="user_embedding.json")
    parser.add_argument('--val_portion', type=float, default=0.2)
    parser.add_argument('--team_member_file', type=str, default="team_members.json")
    parser.add_argument('--user_social_file', type=str, default="user_social.json")
    parser.add_argument('--user_interest_file', type=str, default="user_interests_train.json")
    parser.add_argument('--team_interest_file', type=str, default="team_interests_target.json")
    parser.add_argument('--model_path', type=str, default=None)

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    model = SoAGREE(128,32,0)
    model.load_state_dict(torch.load(args.model_path))
    if args.cuda:
        model.cuda()
    
    loss = nn.BCELoss()
    dataset = GATDataset(args)
    val_loader = dataset.get_data_loader(torch.device('cpu'),['val'])[0]

    del dataset

    avg_loss_val = validate(model,val_loader,loss,args.cuda)