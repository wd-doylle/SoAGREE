'''
Created on Nov 10, 2017
Store parameters

@author: Lianhai Miao
'''

import torch

class Config(object):
    def __init__(self):
        self.path = '../'
        self.user_dataset = {
            'train':self.path + 'user_interests_train.json',
            'val':self.path + 'user_interests_target.json'
        }
        self.group_dataset = {
            'train':self.path + 'team_interests_train.json',
            'val':self.path + 'team_interests_target.json'
        }
        self.user_in_group_path = self.path + 'team_members.json'
        self.follow_in_user_path = self.path + 'user_social.json'
        self.embedding_size = 32
        self.epoch = 10
        self.num_negatives = 10
        self.batch_size = 512
        self.lr = [0.000005, 0.000001, 0.0000005]
        self.drop_ratio = 0.1
        self.topK = 5
        self.balance = 6
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
