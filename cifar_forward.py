from pickletools import optimize
from random import shuffle
from tabnanny import check
import wandb
import re
import torch.nn.functional as F
from utils import data_utils
import torchvision.transforms as transforms
import torchvision
import numpy as np
import argparse
from nets.models import CIFARModel
import copy
import time
from torch import nn, optim
import torch
import sys
import os
import pickle
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from collections import Counter
from size_constrained_clustering import equal
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def prepare_data(args):
    # Data Augmentation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    train_data, validation_data, test_data, T = data_utils.CIFAR10_fed(
        data_path="../data", dataset_list=['noise'], noise_prob=args.noise_prob, downsample=args.downsample)

    train_set = data_utils.BaiscDataset(train_data, transform=train_transform)
    validation_set = data_utils.BaiscDataset(
        validation_data, transform=test_transform)
    test_set = data_utils.BaiscDataset(test_data, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch, shuffle=True, drop_last=True, num_workers=3)
    validation_loader = torch.utils.data.DataLoader(
        validation_set, batch_size=args.batch * 5, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch * 5, shuffle=False, num_workers=2)

    return train_loader, validation_loader, test_loader, T


def train(args, model, train_loader, optimizer, loss_fun, device, T, e, pre_train_e=10, with_T=False):
    model.train()
    num_data = 0
    loss_all = 0
    F_all = None
    y_all = None
    x_all = None
    l_all = None
    if pre_train_e > 0:
        pre_train_thre = pre_train_e
        output_correction = pre_train_thre
    else:
        pre_train_thre = np.inf
        output_correction = np.inf
    train_iter = iter(train_loader)
    for step in range(len(train_iter)):
        optimizer.zero_grad()
        x, y = next(train_iter)
        num_data += y.size(0)
        x = x.to(device).float()
        y = y.to(device).long()
        T = torch.tensor(P, dtype=torch.float).to(device)

        y_train_onehot = torch.zeros((y.shape[0], 10)).to(device)
        y_train_onehot[np.arange(y.shape[0]), y] = 1

        if with_T:
            output = model.forward_with_P(x, T)
        else:
            output, F = model(x)

        l = -1.0 * torch.log(output) * y_train_onehot
        if step == 0:
            F_all = F.detach().cpu()
            y_all = y.detach().cpu()
            x_all = x.detach().cpu()
            l_all = l.detach().cpu()
        else:
            F_all = torch.vstack((F_all, F.detach().cpu()))
            y_all = torch.hstack((y_all, y.detach().cpu()))
            x_all = torch.vstack((x_all, x.detach().cpu()))
            l_all = torch.vstack((l_all, l.detach().cpu()))
        if e < pre_train_thre:  # epoch number
            print('=> Backward...')
            loss = loss_fun(output, y)
            loss.backward()
            loss_all += loss.item()
            optimizer.step()
    if e >= -1:
        print('=> Train finished')
        l_all = torch.sum(l_all, axis=1).reshape(-1, 1)  # loss: nx1
        l_all = l_all.detach().cpu().numpy()
        y_all = y_all.numpy()
        print('=> Start EM')
        gm = GaussianMixture(n_components=2, random_state=2).fit(l_all)
        noisy_pred = gm.predict(l_all)
        pred_score = gm.predict_proba(l_all)
        # gm.means_ 2x1 --> confidence, negative entropy
        smaller_mean = np.min(gm.means_)
        larger_mean = np.max(gm.means_)
        smaller_mean_idx = np.where(gm.means_ == smaller_mean)[0]
        larger_mean_idx = np.where(gm.means_ == larger_mean)[0]
        # get clean index
        clean_idx_candidate = np.where(noisy_pred == smaller_mean_idx)[0]
        clean_idx = []
        max_score = 0.0
        # threhold = 0.99 + (e / 100)
        # top 50%
        for i in clean_idx_candidate:
            if max(pred_score[i]) > 0.99:
                clean_idx.append(i)
            if max(pred_score[i]) > max_score:
                max_score = max(pred_score[i])
        print(max_score)
        noisy_idx = np.setdiff1d(np.arange(0, len(l_all)), np.array(clean_idx))
        print('=> Clean length:', len(clean_idx))
        print('=> Noisy length', len(noisy_idx))
    if e >= pre_train_thre:
        # G = F_all @ F_all.T
        print('=> Start Clustering')
        clustering = equal.SameSizeKMeansMinCostFlow(10)
        clustering.fit(F_all)
        clustering_pred = clustering.labels_
        print(Counter(clustering_pred))
        # clustering = KMeans(n_clusters=10, random_state=0).fit(G)
        # clustering_pred = clustering.predict(G)
        # confidential_score = clustering.predict_proba(G)
        # print(confidential_score)
        # clustering_clean = clustering_pred[clean_idx]
        # clustering_noisy = clustering_pred[noisy_idx]
        # y_clean = y_all[clean_idx]
        # y_noisy = y_all[noisy_idx]
        clean_gt_label = dict()
        cluster_gt = dict()
        cluster_count = dict()
        for i in range(10):
            clean_gt_label[i] = []
            cluster_gt[i] = -1

        for clean_i in clean_idx:
            cl = clustering_pred[clean_i]
            gt = y_all[clean_i]
            clean_gt_label[cl].append(gt)
        for i in range(10):
            c = Counter(clean_gt_label[i])
            print(i, c, c.most_common()[0])
            cluster_gt[i] = c.most_common()[0][0]
            cluster_count[cluster_gt[i]] = sum(c.values())
            
        print(cluster_gt)
        print(cluster_count)
        T = np.zeros((10, 10))
        for cl, y in zip(clustering_pred, y_all):
            col = cluster_gt[cl]
            row = y
            T[row, col] += 1
        
        # for noise_i in noisy_idx:
        #     y_corrupted = y_all[noise_i]
        #     true_y = cluster_gt[clustering_pred[noise_i]]
        #     T[y_corrupted, true_y] += 1
        print(T)
        # for i in range(10):
        #     # noisy_count = np.sum(T[:, i]) - T[i, i]
        #     # T[i, i] = cluster_count[i] - noisy_count
        #     T[:, i] /= cluster_count[i] / 320
        T /= 320
        print(T)
        # T /= (len(clean_idx) + len(noisy_idx))
        # T *= 10
        np.set_printoptions(precision=3)
        print(T)
        print(np.sum(T, axis=0))

        if e >= output_correction:
            train_iter = iter(train_loader)
            for step in range(len(train_iter)):
                optimizer.zero_grad()
                x, y = next(train_iter)
                num_data += y.size(0)
                x = x.to(device).float()
                y = y.to(device).long()
                T = torch.tensor(T, dtype=torch.float).to(device)
                output = model.forward_with_P(x, T)

                loss = loss_fun(output, y)
                loss.backward()
                loss_all += loss.item()
                optimizer.step()

    if e % 20 == 0:
        with open(f'epoch_{e}_F_{args.noise_prob}.pkl', 'wb') as f:
            pickle.dump(F_all, f)
        with open(f'epoch_{e}_y_{args.noise_prob}.pkl', 'wb') as f:
            pickle.dump(y_all, f)
        with open(f'epoch_{e}_x_{args.noise_prob}.pkl', 'wb') as f:
            pickle.dump(x_all, f)
        print(f'=> {e} pickle saved.', F_all.shape, y_all.shape)
    return loss_all / len(train_iter)


def test(model, test_loader, loss_fun, device, classnum=10):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    noises = {}

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device).float()
            target = target.to(device).long()

            output = model.predict(data)

            pred = output.data.max(1)[1]
            correct += pred.eq(target.view(-1)).sum().item()
            total += target.size(0)

    test_error = (total - correct) / total

    return test_error


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device, '\n')
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true',
                        help='test the pretrained model')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--wdecay', type=float,
                        default=0., help='learning rate')
    parser.add_argument('--batch', type=int, default=128, help='batch size')
    parser.add_argument('--epoch', type=int, default=100,
                        help='iterations for communication')
    parser.add_argument('--wk_iters', type=int, default=1,
                        help='optimization iters in local worker between communication')
    parser.add_argument('--save_path', type=str,
                        default='../checkpoint/cifar10', help='path to save the checkpoint')
    parser.add_argument('--resume', action='store_true',
                        help='resume training from the save path checkpoint')
    parser.add_argument('--classnum', type=int, default=10, help='class num')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--noise_prob', type=float, default=0.1)
    parser.add_argument('--pre_train_epoch', type=int, default=-1)
    parser.add_argument('--downsample', action='store_true')

    args = parser.parse_args()
    print(args)

    setup_seed(args.seed)
    exp_folder = 'cifar10_fed_forward'
    args.save_path = os.path.join(args.save_path, exp_folder)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    SAVE_PATH = os.path.join(args.save_path, 'seed' + str(args.seed))
    model = CIFARModel(class_num=args.classnum).to(device)
    loss_fun = nn.CrossEntropyLoss()

    train_loader, validation_loader, test_loader, P = prepare_data(args)
    best_test_error = 1.

    for e in range(args.epoch):
        optimizer = optim.Adam(params=model.parameters(), lr=args.lr)
        print("============ Train epoch {} ============".format(e))
        train_loss = train(args, model, train_loader,
                           optimizer, loss_fun, device, P, e, args.pre_train_epoch)
        print(f'Training Loss: {train_loss}')

        # Testing
        test_loss = test(model, validation_loader, loss_fun,
                         device, classnum=args.classnum)
        print(f'Testing Loss: {test_loss}')

        if test_loss < best_test_error:
            best_test_error = test_loss

            # save the best checkpoint
            print(f'Saving best checkpoint to {SAVE_PATH}')
            torch.save({
                'weight': model.state_dict(),
                'epoch': e,
            }, SAVE_PATH)
        print(
            f'Best Validation Error: {best_test_error}, Current Validation Error: {test_loss}')

    print('Start final testing')
    checkpoint = torch.load(SAVE_PATH)
    model.load_state_dict(checkpoint['weight'])
    test_loss = test(model, test_loader, loss_fun,
                     device, classnum=args.classnum)
    print(f'Best Test Error: {test_loss}')
