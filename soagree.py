'''
Created on Nov 10, 2017
Create Model

@author: Lianhai Miao
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class SoAGREE(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, drop_ratio):
        super(SoAGREE, self).__init__()
        self.followAttention = AttentionLayer((2 * embedding_dim, hidden_dim), drop_ratio)
        self.attention = AttentionLayer((2 * embedding_dim,hidden_dim), drop_ratio)
        self.predictlayer = PredictLayer((3 * embedding_dim,hidden_dim), drop_ratio)

        # initial model
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)


    def forward(self, repo_embed, team_embeds, team_members, user_embeds, user_neighbors):
        # train group
        if team_embeds is not None:
            out = self.grp_forward(repo_embed, team_embeds, team_members, user_embeds, user_neighbors)
        # train user
        else:
            out = self.usr_forward(repo_embed, user_embeds, user_neighbors)
        return out

    # group forward
    def grp_forward(self, repo_embed, team_embeds, team_members, user_embeds, user_neighbors):
        user_aggregates = self.user_aggregate(user_embeds,user_neighbors)
        group_embeds = []
        for team_embed,members in zip(team_embeds,team_members):
            if not members:
                group_embeds.append(team_embed)
                continue
            member_embeds = []
            for mem in members:
                member_embeds.append(user_aggregates[mem])
            member_embeds = torch.stack(member_embeds)
            group_item_embeds = torch.cat((member_embeds, torch.stack([repo_embed]*len(member_embeds))),dim=1)
            at_wt = self.attention(group_item_embeds)
            g_embeds_with_attention = torch.matmul(at_wt, member_embeds)
            g_embeds = g_embeds_with_attention + team_embed
            group_embeds.append(g_embeds.view(-1))

        group_embeds = torch.stack(group_embeds)
        element_embeds = torch.mul(group_embeds, repo_embed)  # Element-wise product
        new_embeds = torch.cat((element_embeds, group_embeds, torch.stack([repo_embed]*len(element_embeds))), dim=1)
        y = torch.sigmoid(self.predictlayer(new_embeds))
        return y

    # user follow aggregate
    def user_aggregate(self, user_embeds, user_neighbors):
        user_finnal_list = []
        for user_embed,neighbors in zip(user_embeds,user_neighbors):
            if not  neighbors:
                user_finnal_list.append(user_embed)
                continue
            neighbor_embeds = torch.stack([user_embeds[i] for i in neighbors])
            user_follow_embeds = torch.cat((neighbor_embeds, torch.stack([user_embed]*len(neighbor_embeds))),dim=1)
            at_wt = self.followAttention(user_follow_embeds)
            u_embeds_with_attention = torch.matmul(at_wt, neighbor_embeds)
            u_embeds = u_embeds_with_attention + user_embed
            user_finnal_list.append(u_embeds.view(-1))
        user_finnal_vec = torch.stack(user_finnal_list, dim=0)
        return user_finnal_vec

    # user forward
    def usr_forward(self, repo_embed, user_embeds, user_neighbors):
        user_aggregates = self.user_aggregate(user_embeds, user_neighbors)
        element_embeds = torch.mul(user_aggregates, repo_embed)  # Element-wise product
        new_embeds = torch.cat((element_embeds, user_aggregates, torch.stack([repo_embed]*len(element_embeds))), dim=1)
        y = torch.sigmoid(self.predictlayer(new_embeds))
        return y

    def tensor2np(self, tens):
        return tens.cpu().numpy()


class AttentionLayer(nn.Module):
    def __init__(self, dims, drop_ratio=0):
        super(AttentionLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(dims[1], 1),
        )

    def forward(self, x):
        out = self.linear(x)
        weight = F.softmax(out.view(1, -1), dim=1)
        return weight


class PredictLayer(nn.Module):
    def __init__(self, dims, drop_ratio=0):
        super(PredictLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(dims[1], 1)
        )

    def forward(self, x):
        out = self.linear(x)
        return out

