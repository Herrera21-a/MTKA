import argparse
import logging
import math
import os
import random
import shutil
import time
import pickle
import numpy as np
from re import search
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
import polaroid
from utils import AverageMeter, accuracy
from utils.utils import *
import pandas as pd
from torchvision import transforms
from torch.autograd import Variable
from models import cnn13_feature as models
from torchvision import utils
from models import AFF
import time



def kdloss(y, teacher_scores):
    p = F.log_softmax(y, dim=1)
    q = F.softmax(teacher_scores, dim=1)
    l_kl = F.kl_div(p, q, reduction="sum") / y.shape[0]
    return l_kl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--scene', type=str, default='./data/')
    parser.add_argument('--n_epochs', type=int, default=60, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=20, help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.03, help='learning rate')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')

    opt = parser.parse_args()
    device = torch.device('cuda', opt.gpu_id)  # 多卡训练只要torch.device('cuda')就好了
    opt.device = device

    model = models.cnn13(num_classes=2, dropout=0).to(opt.device)

    # teacher1 = models.cnn13(num_classes=2)
    # teacher1.load_state_dict(torch.load(opt.scene + "factory1.pth")["state_dict"])
    # teacher2 = models.cnn13(num_classes=2)
    # teacher2.load_state_dict(torch.load(opt.scene + "factory2.pth")["state_dict"])
    # teacher3 = models.cnn13(num_classes=2)
    # teacher3.load_state_dict(torch.load(opt.scene + "factory3.pth")["state_dict"])
    teacher1 = models.cnn13(num_classes=2)
    teacher1.load_state_dict(torch.load("./data/fac_0.1/factory1.pth", map_location={'cuda:5': 'cuda:0'})["state_dict"])
    teacher2 = models.cnn13(num_classes=2)
    teacher2.load_state_dict(torch.load("./data/fac_0.1/factory2.pth", map_location={'cuda:5': 'cuda:0'})["state_dict"])
    teacher3 = models.cnn13(num_classes=2)
    teacher3.load_state_dict(torch.load("./data/fac_0.1/factory3.pth", map_location={'cuda:7': 'cuda:0'})["state_dict"])

    teacher1 = teacher1.to(opt.device)
    teacher1.eval()
    teacher2 = teacher2.to(opt.device)
    teacher2.eval()
    teacher3 = teacher3.to(opt.device)
    teacher3.eval()

    aff_1 = AFF.aff_module(128,4).to(opt.device)
    aff_2 = AFF.aff_module(256,4).to(opt.device)
    aff_3 = AFF.aff_module(512,4).to(opt.device)
    optimizer_a1 = optim.SGD(aff_1.parameters(), lr=opt.lr, momentum=0.9, nesterov=opt.nesterov)
    optimizer_a2 = optim.SGD(aff_2.parameters(), lr=opt.lr, momentum=0.9, nesterov=opt.nesterov)
    optimizer_a3 = optim.SGD(aff_3.parameters(), lr=opt.lr, momentum=0.9, nesterov=opt.nesterov)

    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])

    data_test_1 = polaroid.Polaroid(opt.scene + "factory1", train=False, transform=transform_test)
    data_test_loader_1 = DataLoader(data_test_1, batch_size=50, num_workers=8)
    data_test_2 = polaroid.Polaroid(opt.scene + "factory2", train=False, transform=transform_test)
    data_test_loader_2 = DataLoader(data_test_2, batch_size=50, num_workers=8)
    data_test_3 = polaroid.Polaroid(opt.scene + "factory3", train=False, transform=transform_test)
    data_test_loader_3 = DataLoader(data_test_3, batch_size=50, num_workers=8)

    total_correct = 0
    for k, (images, labels) in enumerate(data_test_loader_1):
        images = images.to(device)
        labels = labels.to(device)
        output = teacher1(images)
        pred = output.data.max(1)[1]
        total_correct += pred.eq(labels.data.view_as(pred)).sum()
    print('Teacher1 Factory1 Accuracy: %f' % (float(total_correct) / len(data_test_1)))
    total_correct = 0
    for k, (images, labels) in enumerate(data_test_loader_2):
        images = images.to(device)
        labels = labels.to(device)
        output = teacher2(images)
        pred = output.data.max(1)[1]
        total_correct += pred.eq(labels.data.view_as(pred)).sum()
    print('Teacher2 Factory2 Accuracy: %f' % (float(total_correct) / len(data_test_2)))
    total_correct = 0
    for k, (images, labels) in enumerate(data_test_loader_3):
        images = images.to(device)
        labels = labels.to(device)
        output = teacher3(images)
        pred = output.data.max(1)[1]
        total_correct += pred.eq(labels.data.view_as(pred)).sum()
    print('Teacher3 Factory3 Accuracy: %f' % (float(total_correct) / len(data_test_3)))

    data_train_1 = polaroid.Polaroid(opt.scene + "factory1", train=True, transform=transform_test)
    data_train_loader_1 = DataLoader(data_train_1, batch_size=20, num_workers=8, shuffle=True)
    data_train_2 = polaroid.Polaroid(opt.scene + "factory2", train=True, transform=transform_test)
    data_train_loader_2 = DataLoader(data_train_2, batch_size=20, num_workers=8, shuffle=True)
    data_train_3 = polaroid.Polaroid(opt.scene + "factory3", train=True, transform=transform_test)
    data_train_loader_3 = DataLoader(data_train_3, batch_size=20, num_workers=8, shuffle=True)

    dt1 = iter(data_train_loader_1)
    dt2 = iter(data_train_loader_2)
    dt3 = iter(data_train_loader_3)

    opt.iteration = 80
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, nesterov=opt.nesterov)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    opt.total_steps = opt.n_epochs * opt.iteration * 1.1
    scheduler = get_cosine_schedule_with_warmup(optimizer, opt.warmup * opt.iteration, opt.total_steps)

    loss_fn = torch.nn.MSELoss()
    model.zero_grad()
    aff_1.zero_grad()
    aff_2.zero_grad()
    aff_3.zero_grad()
    for i in range(opt.n_epochs):
        model.train()
        aff_1.train()
        aff_2.train()
        aff_3.train()
        for j in range(opt.iteration):
            try:
                inputs_1, labels_1 = next(dt1)
            except StopIteration as e:
                dt1 = iter(data_train_loader_1)
                inputs_1, labels_1 = next(dt1)
            try:
                inputs_2, labels_2 = next(dt2)
            except StopIteration as e:
                dt2 = iter(data_train_loader_2)
                inputs_2, labels_2 = next(dt2)
            try:
                inputs_3, labels_3 = next(dt3)
            except StopIteration as e:
                dt3 = iter(data_train_loader_3)
                inputs_3, labels_3 = next(dt3)

            inputs_1 = inputs_1.to(device)
            inputs_2 = inputs_2.to(device)
            inputs_3 = inputs_3.to(device)
            labels_1 = labels_1.to(device)
            labels_2 = labels_2.to(device)
            labels_3 = labels_3.to(device)

            inputs = torch.cat([inputs_1, inputs_2, inputs_3], dim=0)
            output_t1, f_t1 = teacher1(inputs, out_feature=True)
            output_t2, f_t2 = teacher2(inputs, out_feature=True)
            output_t3, f_t3 = teacher3(inputs, out_feature=True)
            label = torch.cat([labels_1, labels_2, labels_3], dim=0)
            f_1 = aff_1(f_t1[0], f_t2[0], f_t3[0])
            f_2 = aff_2(f_t1[1], f_t2[1], f_t3[1])
            f_3 = aff_3(f_t1[2], f_t2[2], f_t3[2])
            logits, f_s = model(inputs, out_feature=True)
            loss_1 = F.cross_entropy(logits, label, reduction='mean')
            loss_2 = loss_fn(f_1, f_s[0]) + loss_fn(f_2, f_s[1]) + loss_fn(f_3, f_s[2])
            loss = loss_1 + 0.001 * loss_2
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer_a1.step()
            optimizer_a2.step()
            optimizer_a3.step()
            model.zero_grad()
            aff_1.zero_grad()
            aff_2.zero_grad()
            aff_3.zero_grad()
            if j == 1:
                print("[curent train loss: % f]" % (loss.item()))
                print("[curent loss_kd: % f   curent loss_feature: % f]" % (loss_1.item(), loss_2.item()))
                print("-------------------------------------------------------")

        with torch.no_grad():
            if (i + 1) % 10 == 0:
                total_correct = 0
                for k, (images, labels) in enumerate(data_test_loader_1):
                    images = images.to(device)
                    labels = labels.to(device)
                    model.eval()
                    output = model(images)
                    pred = output.data.max(1)[1]
                    total_correct += pred.eq(labels.data.view_as(pred)).sum()
                print('Factory1 Test Avg. Accuracy: %f' % (float(total_correct) / len(data_test_1)))
                if (i + 1) == 60:
                    Factory1_acc = float(total_correct) / len(data_test_1)
                total_correct = 0
                for k, (images, labels) in enumerate(data_test_loader_2):
                    images = images.to(device)
                    labels = labels.to(device)
                    model.eval()
                    output = model(images)
                    pred = output.data.max(1)[1]
                    total_correct += pred.eq(labels.data.view_as(pred)).sum()
                print('Factory2 Test Avg. Accuracy: %f' % (float(total_correct) / len(data_test_2)))
                if (i + 1) == 60:
                    Factory2_acc = float(total_correct) / len(data_test_2)
                total_correct = 0
                for k, (images, labels) in enumerate(data_test_loader_3):
                    images = images.to(device)
                    labels = labels.to(device)
                    model.eval()
                    output = model(images)
                    pred = output.data.max(1)[1]
                    total_correct += pred.eq(labels.data.view_as(pred)).sum()
                print('Factory3 Test Avg. Accuracy: %f' % (float(total_correct) / len(data_test_3)))
                if (i + 1) == 60:
                    Factory3_acc = float(total_correct) / len(data_test_3)

    return Factory1_acc, Factory2_acc, Factory3_acc


if __name__ == '__main__':
    cudnn.benchmark = True
    acc_notes = np.zeros([10, 3])
    for exp_num in range(10):
        acc_notes[exp_num, :] = main()
        print(acc_notes[exp_num, :])
    df = pd.DataFrame(acc_notes, columns=["factory1_acc", "factory2_acc", "factory3_acc"])
    df.to_csv(f'{dataset}.csv')




