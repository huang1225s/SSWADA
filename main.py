"""
@author: Huang Yi
"""
from __future__ import print_function
import argparse
import torch.nn as nn
import torch.optim as optim
from utils import *
import torch.nn.functional as F
import os
import time
import torch
import numpy as np
from utils_HSI import sample_gt, metrics, seed_worker
from datasets import get_dataset, HyperX
import torch.utils.data as data
from domain_discriminator import DomainDiscriminator
from iwan import ImportanceWeightModule
from dann import DomainAdversarialLoss
import scipy.io as io
from FeatureNet import Features, ResClassifier

# Training settings
parser = argparse.ArgumentParser(description='SSWADA for Domain Adaptation')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--num_epoch', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--optimizer', type=str, default='momentum', metavar='OP',
                    help='the name of optimizer')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1220, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--patch_size', type=int, default=7,
                    help="Size of the spatial neighbourhood (optional, if "
                         "absent will be set by the model)")
parser.add_argument('--gam', default=0.5, type=float, help='parameter for D and D_0')
parser.add_argument('--la', default=0.1, type=float, help='parameter for dis')
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--num_trials', type=int, default=1,
                    help='the number of epoch')
parser.add_argument('--log_interval', type=int, default=5, metavar='N',
                    help='how many batches to wait before logging training status')
# Data augmentation parameters
group_da = parser.add_argument_group('Data augmentation')
group_da.add_argument('--flip_augmentation', action='store_true', default=True,
                      help="Random flips (if patch_size > 1)")
group_da.add_argument('--radiation_augmentation', action='store_true', default=True,
                      help="Random radiation noise (illumination)")
group_da.add_argument('--mixture_augmentation', action='store_true', default=False,
                      help="Random mixes between spectra")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEV = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
args.cuda = not args.no_cuda and torch.cuda.is_available()
batch_size = args.batch_size
lr = args.lr
use_gpu = torch.cuda.is_available()

if __name__ == '__main__':

 for flag in range(args.num_trials):
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)
    # if args.cuda:
    #     torch.cuda.manual_seed(args.seed)
    seed_worker(args.seed)
    best_acc = 0
    acc_test_list, acc_maxval_test_list = np.zeros_like(args.lr), np.zeros_like(args.lr)
    source_name = 'Hangzhou'
    target_name = 'Shanghai'
    FOLDER = 'E:/huang/PCDA230306/data/DataCube/'
    result_dir = './Result_comparison/'

    img_src, gt_src, LABEL_VALUES_src, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(source_name,
                                                                                        FOLDER)
    img_tar, gt_tar, LABEL_VALUES_tar, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(target_name,
                                                                                        FOLDER)
    num_classes = gt_src.max()
    N_BANDS = img_src.shape[-1]
    hyperparams = vars(args)
    hyperparams.update({'n_classes': num_classes, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS,
                        'device': DEV, 'center_pixel': False, 'supervision': 'full'})
    hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)

    # padding补边
    r = int(hyperparams['patch_size'] / 2) + 1
    img_src = np.pad(img_src, ((r, r), (r, r), (0, 0)), 'symmetric')
    img_tar = np.pad(img_tar, ((r, r), (r, r), (0, 0)), 'symmetric')
    gt_src = np.pad(gt_src, ((r, r), (r, r)), 'constant', constant_values=(0, 0))
    gt_tar = np.pad(gt_tar, ((r, r), (r, r)), 'constant', constant_values=(0, 0))

    train_gt_src, val_gt_src, training_set, valing_set = sample_gt(gt_src, 0.8, mode='random')
    test_gt_tar, _, tesing_set, _ = sample_gt(gt_tar, 1, mode='random')
    train_gt_tar, _, _, _ = sample_gt(gt_tar, 0.5, mode='random')
    img_src_con, img_tar_con, train_gt_src_con, train_gt_tar_con = img_src, img_tar, train_gt_src, train_gt_tar

    # Generate the dataset
    hyperparams_train = hyperparams.copy()
    hyperparams_train.update({'flip_augmentation': True, 'radiation_augmentation': True, 'mixture_augmentation': False})
    train_dataset = HyperX(img_src_con, train_gt_src_con, **hyperparams_train)

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=hyperparams['batch_size'],
                                   pin_memory=True,
                                   # num_workers=4,
                                   shuffle=True,
                                   drop_last=False)
    val_dataset = HyperX(img_src, val_gt_src, **hyperparams)
    val_loader = data.DataLoader(val_dataset,
                                 pin_memory=True,
                                 # num_workers=4,
                                 batch_size=hyperparams['batch_size'],
                                 shuffle=True,
                                 drop_last=False)
    train_tar_dataset = HyperX(img_tar_con, train_gt_tar_con, **hyperparams)
    train_tar_loader = data.DataLoader(train_tar_dataset,
                                       pin_memory=True,
                                       # num_workers=4,
                                       batch_size=hyperparams['batch_size'],
                                       shuffle=True,
                                       drop_last=False)
    test_dataset = HyperX(img_tar, test_gt_tar, **hyperparams)
    test_loader = data.DataLoader(test_dataset,
                                  pin_memory=True,
                                  # num_workers=4,
                                  batch_size=hyperparams['batch_size'],
                                  shuffle=False,
                                  drop_last=False)
    len_src_loader = len(train_loader)
    len_tar_train_loader = len(train_tar_loader)
    len_src_dataset = len(train_loader.dataset)
    len_tar_train_dataset = len(train_tar_loader.dataset)
    len_tar_dataset = len(test_loader.dataset)
    len_tar_loader = len(test_loader)
    len_val_dataset = len(val_loader.dataset)
    len_val_loader = len(val_loader)
    print(hyperparams)

    G = Features(N_BANDS, num_classes, args.patch_size)
    C = ResClassifier(num_classes)
    F1 = ResClassifier(num_classes)
    F2 = ResClassifier(num_classes)
    F1.apply(weights_init)
    F2.apply(weights_init)
    C.apply(weights_init)
    # define domain classifie
    D = DomainDiscriminator(in_feature=2048, hidden_size=1024, batch_norm=False)
    D_0 = DomainDiscriminator(in_feature=2048, hidden_size=1024, batch_norm=False)
    lr = args.lr
    if args.cuda:
        G.cuda()
        C.cuda()
        F1.cuda()
        F2.cuda()
        D.cuda()
        D_0.cuda()
    if args.optimizer == 'momentum':
        optimizer_g = optim.SGD(list(G.parameters()), lr=args.lr, weight_decay=0.0005)
        optimizer_d = optim.SGD(list(D.parameters()), lr=args.lr, weight_decay=0.0005)
        optimizer_d0 = optim.SGD(list(D_0.parameters()), lr=args.lr, weight_decay=0.0005)
        optimizer_f = optim.SGD(list(C.parameters()), momentum=0.9, lr=args.lr, weight_decay=0.0005)
        optimizer_f1f2 = optim.SGD(list(F1.parameters()) + list(F2.parameters()), momentum=0.9,
                                   lr=args.lr, weight_decay=0.0005)
    def reset_grad():
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
        optimizer_f1f2.zero_grad()
        optimizer_d.zero_grad()
        optimizer_d0.zero_grad()

    # define loss function
    domain_adv_D = DomainAdversarialLoss(D).cuda()
    domain_adv_D_0 = DomainAdversarialLoss(D_0).cuda()
    # define importance weight module
    importance_weight_module = ImportanceWeightModule(D, train_dataset.indices)

    def discrepancy(out1, out2):
        return torch.mean(torch.abs(F.softmax(out1) - F.softmax(out2)))

    def train(ep, train_loader, train_tar_loader):
        iter_source, iter_target = iter(train_loader), iter(train_tar_loader)
        criterion = nn.CrossEntropyLoss().cuda()
        G.train()
        C.train()
        D.train()
        D_0.train()
        F1.train()
        F2.train()
        num_iter = len_src_loader
        start_time = time.time()
        for batch_idx in range(1, num_iter):
            if batch_idx % len(train_tar_loader) == 0:
                iter_target = iter(train_tar_loader)
            data_source, label_source = iter_source.next()
            data_target, _ = iter_target.next()
            label_source = label_source - 1
            if args.cuda:
                data1, target1 = data_source.cuda(), label_source.cuda()
                data2 = data_target.cuda()
            # when pretraining network source only
            data = Variable(torch.cat((data1, data2), 0))
            target1 = Variable(target1)

            # Step A train all networks to minimize loss on source
            output = G(data)
            output = C(output)
            output_s = output[:batch_size, :]
            output_t = output[batch_size:, :]
            output_t = F.softmax(output_t)
            entropy_loss = - torch.mean(torch.log(torch.mean(output_t, 0) + 1e-6))

            loss = criterion(output_s, target1.long())
            all_loss = loss + 0.01 * entropy_loss
            all_loss.backward()
            optimizer_g.step()
            reset_grad()

            # Step B
            f_s = G(data1)
            f_t = G(data2)
            adv_loss_D = domain_adv_D(f_s.detach(), f_t.detach(), batch_size).cuda()
            # get importance weights
            w_s = importance_weight_module.get_importance_weight(f_s)
            # domain adversarial loss for D_0
            adv_loss_D_0 = domain_adv_D_0(f_s, f_t, batch_size, w_s=w_s).cuda()
            loss_adv_d = adv_loss_D_0 + adv_loss_D

            output_cr_s_F = C(f_s.cuda())
            output_cr_t_F = C(f_t.cuda())
            output_cr_t_F1 = F1(f_t.cuda())
            output_cr_t_F2 = F2(f_t.cuda())

            loss_cr_s = criterion(output_cr_s_F, target1.long())
            loss_41 = discrepancy(output_cr_t_F1, output_cr_t_F2)
            loss_42 = discrepancy(output_cr_t_F, output_cr_t_F1)
            loss_43 = discrepancy(output_cr_t_F, output_cr_t_F2)
            loss_dis = loss_42 + loss_43 + loss_41
            F_loss = loss_cr_s + args.la * loss_dis + args.gam * loss_adv_d
            F_loss.backward()
            optimizer_f1f2.step()
            optimizer_f.step()
            optimizer_d0.step()
            reset_grad()

            # Step C
            f_s = G(data1)
            f_t = G(data2)
            adv_loss_D = domain_adv_D(f_s.detach(), f_t.detach(), batch_size)
            # get importance weights
            w_s = importance_weight_module.get_importance_weight(f_s)
            # domain adversarial loss for D_0
            adv_loss_D_0 = domain_adv_D_0(f_s, f_t, batch_size, w_s=w_s)
            loss_adv_d = adv_loss_D_0 + adv_loss_D

            output_cr_s_F = C(f_s.cuda())
            output_cr_t_F = C(f_t.cuda())
            output_cr_t_F1 = F1(f_t.cuda())
            output_cr_t_F2 = F2(f_t.cuda())

            lossC = criterion(output_cr_s_F, target1.long())
            loss_41 = discrepancy(output_cr_t_F1, output_cr_t_F2)
            loss_42 = discrepancy(output_cr_t_F, output_cr_t_F1)
            loss_43 = discrepancy(output_cr_t_F, output_cr_t_F2)
            loss_dis = loss_41 + loss_42 + loss_43
            loss = args.la * loss_dis - args.gam * loss_adv_d

            loss.backward()
            optimizer_g.step()
            reset_grad()
            if batch_idx % args.log_interval == 0:
                print(
                    'Train Ep: {} [{}/{} ({:.0f}%)]\t lossC: {:.6f}\t loss_adv: {:.6f}\t Dis: {:.6f} Entropy: {:.6f}'.format(
                        ep, batch_idx * len(data), 256. * len(train_loader),
                            100. * batch_idx / len(train_loader), lossC.item(), loss_adv_d.item(), loss_dis.item(),
                        entropy_loss.item()))
        end_time = time.time()
        print(f'running time is {start_time - end_time} s')

    def val(val_loader):
        G.eval()
        C.eval()
        F1.eval()
        F2.eval()
        correct = 0
        size = 0
        pred_list, label_list = [], []
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(DEV), target.to(DEV)
                target2 = target - 1
                data1, target1 = Variable(data), Variable(target2)
                feat = G(data1)
                output1 = C(feat)
                pred1 = output1.data.max(1)[1]
                k = target1.data.size()[0]
                correct += pred1.eq(target1.data).cpu().sum()
                size += k
                acc = 100. * float(correct) / float(size)
                pred_list.append(pred1.cpu().numpy())
                label_list.append(target1.cpu().numpy())
            print('\nVal set: Accuracy: {}/{} ({:.2f}%)\n'.format(correct, len_val_dataset,acc))
        return acc

    def test(test_loader):
        G.eval()
        C.eval()
        F1.eval()
        F2.eval()
        correct = 0
        size = 0
        pred_list, label_list = [], []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.cuda(), target.cuda()
                target = target - 1
                data1, target1 = Variable(data), Variable(target)
                feat = G(data1)
                output1 = C(feat)
                pred1 = output1.data.max(1)[1]
                k = target1.data.size()[0]
                correct += pred1.eq(target1.data).cpu().sum()
                size += k
                acc = 100. * float(correct) / float(size)
                pred_list.append(pred1.cpu().numpy())
                label_list.append(target1.cpu().numpy())

            print('\nTest set Accuracy: {}/{} ({:.2f}%)\n'.format(correct, len_tar_dataset,
                    100. * correct / len_tar_dataset, acc))
        return acc, pred_list, label_list


    for ep in range(1, args.num_epoch + 1):
        train(ep, train_loader, train_tar_loader)
        val(val_loader)
        if ep % args.log_interval == 0:
            acc, pred_list, label_list = test(test_loader)
            if acc > best_acc:
                best_acc = acc
                results = metrics(np.concatenate(pred_list), np.concatenate(label_list),
                                  ignored_labels=hyperparams['ignored_labels'], n_classes=gt_src.max())
                io.savemat(os.path.join(result_dir,
                                        source_name + '_results_' + target_name + '.mat'),
                           {'lr': args.lr, 'la': args.la, 'gam': args.gam, 'results': results})
                print('current best acc:', best_acc)
    print('current best acc:', best_acc)


