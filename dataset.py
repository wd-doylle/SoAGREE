'''
Created on Aug 8, 2016
Processing datasets.

@author: Xiangnan He (xiangnanhe@gmail.com)

Modified  on Nov 10, 2017, by Lianhai Miao
'''

import scipy.sparse as sp
import numpy as np
import torch
import json
from torch.utils.data import TensorDataset, DataLoader

class GDataset(object):

    def __init__(self, config, num_negatives):
        '''
        Constructor
        '''
        self.num_negatives = num_negatives
        self.device = config.device
        self.num_users, self.num_repos, self.num_teams = 0, 0, 0
        # user data
        self.user_trainMatrix = self.load_interest_matrix(config.user_dataset['train'],"user")
        self.user_valMatrix = self.load_interest_matrix(config.user_dataset['val'],"user")
        self.user_dataloader_train = self.get_dataloader(config.batch_size,['user','train'])
        self.user_dataloader_val = self.get_dataloader(config.batch_size,['user','val'])
        # group data
        self.group_trainMatrix = self.load_interest_matrix(config.group_dataset['train'],"team")
        self.group_valMatrix = self.load_interest_matrix(config.group_dataset['val'],"team")
        self.group_dataloader_train = self.get_dataloader(config.batch_size,['team','val'])
        self.group_dataloader_val = self.get_dataloader(config.batch_size,['team','val'])
        # list of members in a team
        self.team_member_dict = self.load_team_member_dict(config.user_in_group_path)
        # social connections of users
        self.user_social_dict = self.load_user_social_dict(config.follow_in_user_path)


    def load_team_member_dict(self, path):
        team_members = {}
        with open(path) as f:
            for l in f.readlines():
                line = json.loads(l)
                team_members[line['team']] = line['members']

        return team_members

    def load_user_social_dict(self, path):
        with open(path) as nj:
            gg = json.load(nj)
            graph = {}
            for n1 in gg:
                graph[int(n1)] = []
                for n2 in gg[n1]:
                    graph[int(n1)].append(int(n2))

        return graph

    def load_interest_matrix(self, filename, key):
        interest = {}
        with open(filename) as tmj:
            for tml in tmj.readlines():
                line = json.loads(tml)
                interest[line[key]] = {int(r):line['interests'][r] for r in line['interests']}
                if key == 'user':
                    self.num_users = max(self.num_users,line[key]+1)
                elif key == 'team':
                    self.num_teams = max(self.num_teams,line[key]+1)
        for user in interest:
            total_contri = 0
            for repo in interest[user]:
                self.num_repos = max(self.num_repos,repo+1)
                total_contri += interest[user][repo]
            for repo in interest[user]:
                interest[user][repo] /= total_contri
        
        return interest

    def get_train_instances(self, train):
        user_input, pos_item_input, neg_item_input = [], [], []
        # num_users = len(train)
        num_repos = self.num_repos
        for u in train:
            for i in train[u]:
                # positive instance
                for _ in range(self.num_negatives):
                    pos_item_input.append(i)
                # negative instances
                for _ in range(self.num_negatives):
                    j = np.random.randint(num_repos)
                    while (u, j) in train:
                        j = np.random.randint(num_repos)
                    user_input.append(u)
                    neg_item_input.append(j)
        pi_ni = [[pi, ni] for pi, ni in zip(pos_item_input, neg_item_input)]
        return user_input, pi_ni

    def get_dataloader(self, batch_size, selector):
        if selector == ['user','train']:
            usr_tm, positem_negitem = self.get_train_instances(self.user_trainMatrix)
        elif selector == ['user','val']:
            usr_tm, positem_negitem = self.get_train_instances(self.user_valMatrix)
        elif selector == ['team','train']:
            usr_tm, positem_negitem = self.get_train_instances(self.group_trainMatrix)
        elif selector == ['team','val']:
            usr_tm, positem_negitem = self.get_train_instances(self.group_valMatrix)
        train_data = TensorDataset(torch.tensor(usr_tm,device=self.device), torch.tensor(positem_negitem,device=self.device))
        data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        return data_loader






