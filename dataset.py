import torch
import json
import argparse
from torch.utils.data import TensorDataset, DataLoader




class GATDataset():

    def __init__(self,config):
        self.repo_features = {}
        with open(config.repo_feature_file) as tmj:
            for rp,l in enumerate(tmj.readlines()):
                line = json.loads(l)
                self.repo_features[rp] = line


        self.T = int((1-config.val_portion)*len(self.repo_features))
        
        self.team_features = []
        with open(config.team_feature_file) as tmj:
            for l in tmj.readlines():
                line = json.loads(l)
                self.team_features.append(line)
        
        self.user_features = []
        with open(config.user_feature_file) as tmj:
            for l in tmj.readlines():
                line = json.loads(l)
                self.user_features.append(line)

        self.team_users = []
        user_team = {}
        with open(config.team_member_file) as tmj:
            for l in tmj.readlines():
                line = json.loads(l)
                self.team_users.append(line['members'])
                for user in line['members']:
                    if not user in user_team:
                        user_team[user] = []
                    user_team[user].append(line['team'])
        
        self.user_social = {}
        with open(config.user_social_file) as tmj:
            u_s = json.load(tmj)
            for u1 in u_s:
                if not int(u1) in self.user_social:
                    self.user_social[int(u1)] = {}
                for u2 in u_s[u1]:
                    self.user_social[int(u1)][int(u2)] = u_s[u1][u2]
                    if not int(u2) in self.user_social:
                        self.user_social[int(u2)] = {}
                    self.user_social[int(u2)][int(u1)] = u_s[u1][u2]

        self.repo_core_users = {}
        with open(config.user_interest_file) as tmj:
            for l in tmj.readlines():
                line = json.loads(l)
                for repo in line['interests']:
                    if not int(repo) in self.repo_core_users:
                        self.repo_core_users[int(repo)] = []
                    self.repo_core_users[int(repo)].append(line['user'])
        
        self.repo_users = {}
        for repo in self.repo_core_users:
            self.repo_users[repo] = set(self.repo_core_users[repo])
            for user in self.repo_core_users[repo]:
                self.repo_users[repo].update(self.user_social[user])
        
        self.repo_teams_one = {}
        with open(config.team_interest_file) as tmj:
            for l in tmj.readlines():
                line = json.loads(l)
                for repo in line['interests']:
                    if not int(repo) in self.repo_teams_one:
                        self.repo_teams_one[int(repo)] = set()
                    self.repo_teams_one[int(repo)].add(line['team'])
        self.repo_teams_zero = {}
        for repo in self.repo_users:
            self.repo_teams_zero[repo] = set()
            for user in self.repo_users[repo]:
                if not user in user_team:
                    continue
                if not user in self.repo_core_users[repo]:
                    for team in user_team[user]:
                        if team in self.repo_teams_one[repo]:
                            continue
                        self.repo_teams_zero[repo].add(team)
        self.repo_teams_one = {k:list(self.repo_teams_one[k]) for k in self.repo_teams_one}
        self.repo_teams_zero = {k:list(self.repo_teams_zero[k]) for k in self.repo_teams_zero}

    def get_data_loader(self, device, stages):
        data = []
        for stage in stages:
            data.append([])
            data_range = range(self.T) if stage == 'train' else range(self.T,len(self.repo_features))
            for rp in data_range:
                user_inds = {u:i for i,u in enumerate(self.repo_users[rp])}
                repo = torch.tensor(self.repo_features[rp],device=device)
                users = torch.tensor([self.user_features[u] for u in self.repo_users[rp]],device=device)
                target_users = [0]*len(users)
                for u in self.repo_core_users[rp]:
                    target_users[user_inds[u]] = 1
                target_users = torch.tensor(target_users,dtype=torch.float,device=device).view(-1,1)
                users = torch.tensor([self.user_features[u] for u in self.repo_users[rp]],device=device)
                user_neighbors = [[user_inds[uu] for uu in self.user_social[u] if uu in user_inds] for u in self.repo_users[rp] if u in self.user_social]
                nm_ones = len(self.repo_teams_one[rp])
                nm_zeros = len(self.repo_teams_zero[rp])
                # one_compensate = max(1,nm_zeros//nm_ones) if stage == 'train' else 1
                teams = torch.tensor([self.team_features[tm] for tm in self.repo_teams_zero[rp]+self.repo_teams_one[rp]],device=device)
                team_users = [[user_inds[u] for u in self.team_users[tm] if u in user_inds] for tm in self.repo_teams_zero[rp]+self.repo_teams_one[rp]]
                target_team = torch.tensor([0]*nm_zeros+[1]*nm_ones,dtype=torch.float,device=device).view(-1,1)
                data[-1].append([repo,users,user_neighbors,teams,team_users,target_users, target_team])

        self.repo_features = None
        self.repo_core_users = None
        self.repo_users = None


        return data


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--repo_feature_file', type=str, default="repo_embedding.json")
    parser.add_argument('--team_feature_file', type=str, default="team_embedding.json")
    parser.add_argument('--user_feature_file', type=str, default="user_embedding.json")
    parser.add_argument('--val_portion', type=float, default=0.2)
    parser.add_argument('--team_member_file', type=str, default="team_members.json")
    parser.add_argument('--user_social_file', type=str, default="user_social.json")
    parser.add_argument('--user_interest_file', type=str, default="user_interests_train.json")
    parser.add_argument('--team_interest_file', type=str, default="team_interests_target.json")
    args = parser.parse_args()
    dataset = GATDataset(args)
    
    train_loader,val_loader = dataset.get_data_loader(torch.device('cpu'),['train','val'])
    print(train_loader[0])
