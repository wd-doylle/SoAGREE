'''
Created on Nov 10, 2017
Main function

@author: Lianhai Miao
'''

from model.soagree import SoAGREE
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from time import time
from config import Config
# from utils.util import Helper
from dataset import GDataset


# train the model
def training(model, train_loader, epoch_id, config, type_m):
    # user trainning
    learning_rates = config.lr
    # learning rate decay
    lr = learning_rates[0]
    if epoch_id >= 20 and epoch_id < 50:
        lr = learning_rates[1]
    elif epoch_id >=50:
        lr = learning_rates[2]
    # lr decay
    # if epoch_id % 5 == 0:
    #     lr /= 2

    # optimizer
    optimizer = optim.RMSprop(model.parameters(), lr)

    losses = []
    for batch_id, (u, pi_ni) in enumerate(train_loader):
        # Data Load
        user_input = u
        pos_item_input = pi_ni[:, 0]
        neg_item_input = pi_ni[:, 1]
        # Forward
        if type_m == 'user':
            pos_prediction = model(None, user_input, pos_item_input)
            neg_prediction = model(None, user_input, neg_item_input)
        elif type_m == 'group':
            pos_prediction = model(user_input, None, pos_item_input)
            neg_prediction = model(user_input, None, neg_item_input)
        # Zero_grad
        model.zero_grad()
        # Loss
        loss = torch.mean((pos_prediction - neg_prediction -1) **2)
        # record loss history
        # Backward
        loss.backward()
        optimizer.step()
        losses.append(loss)

    print('Iteration %d, %s loss is [%.4f ]' % (epoch_id, type_m, torch.mean(torch.tensor(losses))))


def validate(model, val_loader, epoch_id, type_m):
    model.eval()
    losses = []
    for batch_id, (u, pi_ni) in enumerate(val_loader):
        # Data Load
        user_input = u
        pos_item_input = pi_ni[:, 0]
        neg_item_input = pi_ni[:, 1]
        # Forward
        if type_m == 'user':
            pos_prediction = model(None, user_input, pos_item_input)
            neg_prediction = model(None, user_input, neg_item_input)
        elif type_m == 'group':
            pos_prediction = model(user_input, None, pos_item_input)
            neg_prediction = model(user_input, None, neg_item_input)
        # Loss
        loss = torch.mean((pos_prediction - neg_prediction -1) **2)
        losses.append(loss)

    print('Iteration %d, %s validation loss is [%.4f ]' % (epoch_id, type_m, torch.mean(torch.tensor(losses))))
    model.train()


def evaluation(model, helper, testRatings, testNegatives, K, type_m):
    model.eval()
    (hits, ndcgs) = helper.evaluate_model(model, testRatings, testNegatives, K, type_m)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    return hr, ndcg


if __name__ == '__main__':
    # initial parameter class
    config = Config()

    # initial helper
    # helper = Helper()

    # initial dataSet class
    print("Loading dataset...")
    td = time()
    dataset = GDataset(config, config.num_negatives)
    print("Loading time is: [%.1f s]" % (time()-td))

    # get the dict of users in group
    g_m_d = dataset.team_member_dict

    # get the dict of follow in user
    u_f_d = dataset.user_social_dict

    # get group number
    num_group, num_users, num_items = dataset.num_teams, dataset.num_users, dataset.num_repos

    # build AGREE model
    soagree = SoAGREE(num_users, num_items, num_group, config.embedding_size, g_m_d, u_f_d, config.drop_ratio,config.device)

    # config information
    print("SoAGREE at embedding size %d, run Iteration:%d, NDCG and HR at %d" %(config.embedding_size, config.epoch, config.topK))
    # train the model
    for epoch in range(config.epoch):
        soagree.train()
        # training start time
        tt = time()

        for _ in range(config.balance):
            training(soagree, dataset.group_dataloader_train, epoch, config, 'group')

        training(soagree, dataset.user_dataloader_train, epoch, config, 'user')

        print("user and group training time is: [%.1f s]" % (time()-tt))
        # validation
        tv = time()
        
        validate(soagree,dataset.user_dataloader_val, epoch, 'user')

        validate(soagree,dataset.group_dataloader_val, epoch, 'group')

        print("Validation time is: [%.1f s]" % (time()-tv))

        #evaluation

        # u_hr, u_ndcg = evaluation(soagree, helper, dataset.user_testRatings, dataset.user_testNegatives, config.topK, 'user')
        # print('User Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, [%.1f s]' % (
            # epoch, time() - t1, u_hr, u_ndcg, time() - t2))

        # hr, ndcg = evaluation(soagree, helper, dataset.group_testRatings, dataset.group_testNegatives, config.topK, 'group')
        # print(
            # 'Group Iteration %d [%.1f s]: HR = %.4f, '
            # 'NDCG = %.4f, [%.1f s]' % (epoch, time() - t1, hr, ndcg, time() - t2))









