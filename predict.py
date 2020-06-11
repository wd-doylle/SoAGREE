import time
import argparse
import numpy as np
import json
import torch
import time

from soagree import SoAGREE
from dataset import GATDataset


def sort_to_k(ary,k,key=lambda x:x,reversed=False):
    k = min(k,len(ary))
    for i in range(k):
        for j in range(len(ary)-1-i):
            if not reversed:
                if key(ary[len(ary)-1-j]) < key(ary[len(ary)-2-j]):
                    ary[len(ary)-1-j],ary[len(ary)-2-j] = ary[len(ary)-2-j],ary[len(ary)-1-j]
            else:
                if key(ary[len(ary)-1-j]) > key(ary[len(ary)-2-j]):
                    ary[len(ary)-1-j],ary[len(ary)-2-j] = ary[len(ary)-2-j],ary[len(ary)-1-j]
    return ary


def predict(model,data_loader,repos,repo_teams,out_file,ks,cuda):
    t = time.time()
    model.eval()
    fs = [open(out_file+'_%d.json'%k,'w') for k in ks]
    for batch_id, (repo,users,user_neighbors,teams,team_users,target_users, target_team) in enumerate(data_loader):
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
        output = output.detach().cpu().numpy()
        recs = sort_to_k(list(range(len(output))),ks[-1],key=lambda i:output[i],reversed=True)
        for i,k in enumerate(ks):
            f = fs[i]
            f.write(str(repos[batch_id]))
            for rec in recs[:k]:
                f.write("\t%s"%json.dumps((repo_teams[batch_id][rec],output[rec].item())))
            f.write('\n')
        if batch_id%1000 == 0:
            print('Batch %d'%batch_id)
    print("Prediction Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--val_portion', type=float, default=0.2)
    parser.add_argument('--model_path', type=str, default="gat_1.pth")
    parser.add_argument('--repo_feature_file', type=str, default="repo_embedding.json")
    parser.add_argument('--team_feature_file', type=str, default="team_embedding.json")
    parser.add_argument('--user_feature_file', type=str, default="user_embedding.json")
    parser.add_argument('--team_member_file', type=str, default="team_members.json")
    parser.add_argument('--user_social_file', type=str, default="user_social.json")
    parser.add_argument('--user_interest_file', type=str, default="user_interests_train.json")
    parser.add_argument('--team_interest_file', type=str, default="team_interests_target.json")
    parser.add_argument('--out_file', type=str, default="soagree_team_score")

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    ks = [5,10,30,50]

    model = SoAGREE(128,32,0)
                
    model.load_state_dict(torch.load(args.model_path))
    if args.cuda:
        model.cuda()

    dataset = GATDataset(args)
    val_loader = dataset.get_data_loader(torch.device('cpu'),['val'])[0]

    T = dataset.T
    N = T+len(val_loader)
    repo_teams = [dataset.repo_teams_zero[r]+dataset.repo_teams_one[r] for r in range(T,N)]
    predict(model,val_loader,list(range(T,N)),repo_teams,args.out_file,ks,args.cuda)
