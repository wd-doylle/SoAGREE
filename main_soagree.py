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
from validate import validate

# Training settings

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=10, help='Patience')
parser.add_argument('--repo_feature_file', type=str, default="repo_embedding.json")
parser.add_argument('--team_feature_file', type=str, default="team_embedding.json")
parser.add_argument('--user_feature_file', type=str, default="user_embedding.json")
parser.add_argument('--val_portion', type=float, default=0.2)
parser.add_argument('--team_member_file', type=str, default="team_members.json")
parser.add_argument('--user_social_file', type=str, default="user_social.json")
parser.add_argument('--user_interest_file', type=str, default="user_interests_train.json")
parser.add_argument('--team_interest_file', type=str, default="team_interests_target.json")
parser.add_argument('--pretrained', type=str, default=None)
parser.add_argument('--out_file', type=str, default="soagree_team_score")
parser.add_argument('--balance', type=int, default=1, help='Alternative architectures')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)




model = SoAGREE(128,128,args.dropout)
if args.pretrained:
    model.load_state_dict(torch.load(args.pretrained))
if args.cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), 
                       lr=args.lr, 
                       weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=2)
loss = nn.BCELoss()
dataset = GATDataset(args)
train_loader,val_loader = dataset.get_data_loader(torch.device('cpu'),['train','val'])
T = dataset.T
N = T+len(val_loader)
repo_teams = [dataset.repo_teams_zero[r]+dataset.repo_teams_one[r] for r in range(T,N)]

del dataset

def train(epoch,train_loader,val_loader,type_m,fastmode=True):
    t = time.time()
    model.eval()
    losses_train = []
    for batch_id, (repo,users,user_neighbors,teams,team_users,target_users, target_team) in enumerate(train_loader):
        optimizer.zero_grad()
        if args.cuda:
            try:
                repo = repo.cuda()
                users = users.cuda()
                user_edges = user_edges.cuda()
                teams = teams.cuda()
                target = target.cuda()
            except:
                continue
        if type_m == 'user':
            output = model(repo, None, None, users, user_neighbors)
            loss_train = loss(output,target_users)
        elif type_m == 'team':
            output = model(repo, teams, team_users, users, user_neighbors)
            loss_train = loss(output,target_team)

        losses_train.append(loss_train.item())
        loss_train.backward()
        optimizer.step()

        if batch_id % 1000 == 0:
            print('Batch %d, loss is [%.8f ]' % (batch_id, losses_train[-1]))

    avg_loss_train = torch.mean(torch.tensor(losses_train))

    print('Epoch: {:04d}'.format(epoch),
          'Phase:%s'%type_m,  
          'loss_train: {:.8f}'.format(avg_loss_train),
          'time: {:.4f}s'.format(time.time() - t))
    if not fastmode:
        return validate(model,val_loader,loss,args.cuda)
    else:
        return avg_loss_train


# Train model
t_total = time.time()
bad_counter = 0
best = 100
best_epoch = 0
for epoch in range(1,args.epochs+1):
    for _ in range(args.balance):
        train(epoch,train_loader,val_loader,'team')
    avg_loss_val = train(epoch,train_loader,val_loader,'user',fastmode=False)

    scheduler.step(avg_loss_val)

    torch.save(model.state_dict(), 'soagree_{}.pth'.format(epoch))
    if best_epoch <= 0 or avg_loss_val < best:
        best = avg_loss_val
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

ks = [5,10,30,50]
predict(model,val_loader,list(range(T,N)),repo_teams,args.out_file,ks,args.cuda)