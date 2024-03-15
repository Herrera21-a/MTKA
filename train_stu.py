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
from utils.test_teacher import test_teacher1,test_teacher2,test_teacher3
import pandas as pd
from torchvision import transforms
from torch.autograd import Variable
from models import cnn13_feature as models
from models import cnn13_stu as models_stu
from torchvision import utils
from models import attention
from models import generator as gen
import numpy
import time

def kdloss(y, teacher_scores):
    p = F.log_softmax(y, dim=1)
    q = F.softmax(teacher_scores, dim=1)
    l_kl = F.kl_div(p, q, reduction="sum") / y.shape[0]
    return l_kl


def loss2weight(output1, output2, output3, label):
    loss_fn = torch.nn.CrossEntropyLoss()
    loss_1 = loss_fn(output1, label)
    loss_2 = loss_fn(output2, label)
    loss_3 = loss_fn(output3, label)
    loss_all = torch.tensor([loss_1, loss_2, loss_3])
    return loss_all


# 定义测试函数




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--data', type=str, default='./data/')
    parser.add_argument('--n_epochs', type=int, default=60, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=400, help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.03, help='learning rate')
    parser.add_argument('--warmup', default=0, type=float,
                            help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--latent_dim', type=int, default=128, help='dimensionality of the latent space')


    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)  # 为CPU设置随机种子
    torch.cuda.manual_seed(1)  # 为所有的GPU设置随机种子

    opt = parser.parse_args()
    device = torch.device('cuda', opt.gpu_id)  # 多卡训练只要torch.device('cuda')就好了
    opt.device = device

    z = torch.randn(opt.batch_size, opt.latent_dim).to(opt.device)

    gen_good_1 = gen.generator().to(opt.device)
    gen_bad_1 = gen.generator().to(opt.device)
    gen_good_2 = gen.generator().to(opt.device)
    gen_bad_2 = gen.generator().to(opt.device)
    gen_good_3 = gen.generator().to(opt.device)
    gen_bad_3 = gen.generator().to(opt.device)

    gen_good_1.load_state_dict(torch.load("./gen_para/factory1_gen_good.pth", map_location='cuda:0')["state_dict"])
    gen_bad_1.load_state_dict(torch.load("./gen_para/factory1_gen_bad.pth", map_location='cuda:0')["state_dict"])
    gen_good_2.load_state_dict(torch.load("./gen_para/factory2_gen_good.pth", map_location='cuda:0')["state_dict"])
    gen_bad_2.load_state_dict(torch.load("./gen_para/factory2_gen_bad.pth", map_location='cuda:0')["state_dict"])
    gen_good_3.load_state_dict(torch.load("./gen_para/factory3_gen_good.pth", map_location='cuda:0')["state_dict"])
    gen_bad_3.load_state_dict(torch.load("./gen_para/factory3_gen_bad.pth", map_location='cuda:0')["state_dict"])

    model = models_stu.cnn13(num_classes=2, dropout=0).to(opt.device)

    # 加载教师模型

    teacher1 = models.cnn13(num_classes=2)
    teacher1.load_state_dict(torch.load("./data/fac_0.1/factory1.pth", map_location={'cuda:5': 'cuda:0'})["state_dict"])
    teacher2 = models.cnn13(num_classes=2)
    teacher2.load_state_dict(torch.load("./data/fac_0.1/factory2.pth", map_location={'cuda:5': 'cuda:0'})["state_dict"])
    teacher3 = models.cnn13(num_classes=2)
    teacher3.load_state_dict(torch.load("./data/fac_0.1/factory3.pth", map_location={'cuda:7': 'cuda:0'})["state_dict"])

    teacher_models=[]
    teacher1 = teacher1.to(opt.device)
    teacher1.eval()
    teacher_models.append(teacher1)
    teacher2 = teacher2.to(opt.device)
    teacher2.eval()
    teacher_models.append(teacher2)
    teacher3 = teacher3.to(opt.device)
    teacher3.eval()
    teacher_models.append(teacher3)


    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])

    data_train_1 = polaroid.Polaroid(opt.data + "factory1", train=True, transform=transform_test)
    data_train_loader_1 = DataLoader(data_train_1, batch_size=20, num_workers=8, shuffle=True)
    data_train_2 = polaroid.Polaroid(opt.data + "factory2", train=True, transform=transform_test)
    data_train_loader_2 = DataLoader(data_train_2, batch_size=20, num_workers=8, shuffle=True)
    data_train_3 = polaroid.Polaroid(opt.data + "factory3", train=True, transform=transform_test)
    data_train_loader_3 = DataLoader(data_train_3, batch_size=20, num_workers=8, shuffle=True)

    data_test_1 = polaroid.Polaroid(opt.data + "factory1", train=False, transform=transform_test)
    data_test_loader_1 = DataLoader(data_test_1, batch_size=50, num_workers=8)
    data_test_2 = polaroid.Polaroid(opt.data + "factory2", train=False, transform=transform_test)
    data_test_loader_2 = DataLoader(data_test_2, batch_size=50, num_workers=8)
    data_test_3 = polaroid.Polaroid(opt.data + "factory3", train=False, transform=transform_test)
    data_test_loader_3 = DataLoader(data_test_3, batch_size=50, num_workers=8)

    opt.iteration = 800 // 20
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, nesterov=opt.nesterov)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    opt.total_steps = opt.n_epochs * opt.iteration * 1.1
    scheduler = get_cosine_schedule_with_warmup(optimizer, opt.warmup * opt.iteration, opt.total_steps)



    img_index = list(range(800))
    gen_good_1.eval()
    gen_bad_1.eval()
    gen_good_2.eval()
    gen_bad_2.eval()
    gen_good_3.eval()
    gen_bad_3.eval()

    gen_imgs_good_1 = gen_good_1(z)
    gen_imgs_bad_1 = gen_bad_1(z)
    gen_imgs_1 = torch.cat([gen_imgs_good_1, gen_imgs_bad_1], dim=0).detach()
    label_1 = torch.zeros(800, dtype=torch.long)
    label_1[400:800] = 1

    gen_imgs_good_2 = gen_good_2(z)
    gen_imgs_bad_2 = gen_bad_2(z)
    gen_imgs_2 = torch.cat([gen_imgs_good_2, gen_imgs_bad_2], dim=0).detach()
    label_2 = torch.zeros(800, dtype=torch.long)
    label_2[400:800] = 1

    gen_imgs_good_3 = gen_good_3(z)
    gen_imgs_bad_3 = gen_bad_3(z)
    gen_imgs_3 = torch.cat([gen_imgs_good_3, gen_imgs_bad_3], dim=0).detach()
    label_3 = torch.zeros(800, dtype=torch.long)
    label_3[400:800] = 1


    dt1 = iter(data_train_loader_1)
    dt2 = iter(data_train_loader_2)
    dt3 = iter(data_train_loader_3)
    model.zero_grad()
    loss_fn = torch.nn.MSELoss()
    select_arr = np.zeros([60, 3])


    for i in range(opt.n_epochs):
        model.train()
        if i<10:
            teacher_accs1 = []
            for teacher_model in teacher_models:
                teacher_acc1 = test_teacher1(model, teacher_model)
                teacher_accs1.append(teacher_acc1)
            best_teacher_index1 = torch.argmax(torch.tensor(teacher_accs1))
            best_teacher_model1 = teacher_models[best_teacher_index1]
            teacher_accs2 = []
            for teacher_model in teacher_models:
                teacher_acc2 = test_teacher2(model, teacher_model, best_teacher_model1)
                teacher_accs2.append(teacher_acc2)
            best_teacher_index2 = torch.argmax(torch.tensor(teacher_accs2))
            best_teacher_model2 = teacher_models[best_teacher_index2]
            teacher_accs3 = []
            for teacher_model in teacher_models:
                teacher_acc3 = test_teacher3(model, teacher_model, best_teacher_model1, best_teacher_model2)
                teacher_accs3.append(teacher_acc3)
            best_teacher_index3 = torch.argmax(torch.tensor(teacher_accs3))
            best_teacher_model3 = teacher_models[best_teacher_index3]
        for j in range(40):
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

            # inputs_1 = gen_imgs_1[j * 20: (j + 1) * 20, :, :, :].to(device)
            # inputs_2 = gen_imgs_2[j * 20: (j + 1) * 20, :, :, :].to(device)
            # inputs_3 = gen_imgs_3[j * 20: (j + 1) * 20, :, :, :].to(device)
            inputs = torch.cat([inputs_1, inputs_2, inputs_3], dim=0)
            output_t1, feature_t1 = teacher1(inputs, out_feature=True)
            output_t2, feature_t2 = teacher2(inputs, out_feature=True)
            output_t3, feature_t3 = teacher3(inputs, out_feature=True)
            weight_1 = loss2weight(output_t1[0:20, :].clone(), output_t2[0:20, :].clone(),
                                   output_t3[0:20, :].clone(), labels_1)
            weight_2 = loss2weight(output_t1[20:40, :].clone(), output_t2[20:40, :].clone(),
                                   output_t3[20:40, :].clone(), labels_2)
            weight_3 = loss2weight(output_t1[40:60, :].clone(), output_t2[40:60, :].clone(),
                                   output_t3[40:60, :].clone(), labels_3)

            # feature1_all = [feature_t1[0:20, :], feature_t2[0:20, :], feature_t3[0:20, :]]
            # feature2_all = [feature_t1[20:40, :], feature_t2[20:40, :], feature_t3[20:40, :]]
            # feature3_all = [feature_t1[40:60, :], feature_t2[40:60, :], feature_t3[40:60, :]]
            feature1_all = [feature_t1[0], feature_t2[0], feature_t3[0]]
            feature2_all = [feature_t1[1], feature_t2[1], feature_t3[1]]
            feature3_all = [feature_t1[2], feature_t2[2], feature_t3[2]]
            feature1_index = torch.min(weight_1, 0)[1]
            feature2_index = torch.min(weight_2, 0)[1]
            feature3_index = torch.min(weight_3, 0)[1]

            if j == 0:
                # print(feature1_index, feature2_index, feature3_index)
                select_arr[i, :] = [feature1_index, feature2_index, feature3_index]

            feature1_final = feature1_all[feature1_index]
            feature2_final = feature2_all[feature2_index]
            feature3_final = feature3_all[feature3_index]
            # feature_t = torch.cat([feature1_all[feature1_index], feature2_all[feature2_index],
            #                        feature3_all[feature3_index]], dim=0)
            kd_target = torch.cat([output_t1[0:20, :], output_t2[20:40, :], output_t3[40:60, :]], dim=0)

            if i < 10:
                # print(i)
                output1, feature1 = best_teacher_model1(inputs, out_feature=True)
                output2, feature2 = best_teacher_model2(inputs, out_feature=True)
                output3, feature3 = best_teacher_model3(inputs, out_feature=True)
                f11 = feature1[0]
                f22 = feature2[1]
                f33 = feature3[2]
                logits, feature_s = model(inputs, f11=f11,f22=f22,f33=f33,out_feature=True, teaching1=True,teaching2=True,teaching3=True)
            else:
                logits, feature_s = model(inputs, f11=f11,f22=f22,f33=f33,out_feature=True, teaching1=False,teaching2=False,teaching3=False)


            loss_feature = loss_fn(feature1_final, feature_s[0]) + loss_fn(feature2_final, feature_s[1]) + loss_fn(feature3_final, feature_s[2])
            loss_kd = kdloss(logits, kd_target)
            loss = loss_feature + loss_kd
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            if j == 1:
                print("[curent train loss: % f]" % (loss.item()))
                print("-------------------------------------------------------")

        with torch.no_grad():
            if (i + 1) % 10 == 0:
                total_correct = 0
                for k, (images, labels) in enumerate(data_test_loader_1):
                    images = images.to(device)
                    labels = labels.to(device)
                    model.eval()
                    output = model(images, f11=f11,f22=f22,f33=f33,out_feature=False, teaching1=False,teaching2=False,teaching3=False)
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
                    output = model(images, f11=f11,f22=f22,f33=f33,out_feature=False, teaching1=False,teaching2=False,teaching3=False)
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
                    output = model(images, f11=f11,f22=f22,f33=f33,out_feature=False, teaching1=False,teaching2=False,teaching3=False)
                    pred = output.data.max(1)[1]
                    total_correct += pred.eq(labels.data.view_as(pred)).sum()
                print('Factory3 Test Avg. Accuracy: %f' % (float(total_correct) / len(data_test_3)))
                if (i + 1) == 60:
                    Factory3_acc = float(total_correct) / len(data_test_3)

    return Factory1_acc, Factory2_acc, Factory3_acc

if __name__ == '__main__':
    cudnn.benchmark = True
    acc_notes = np.zeros([10, 3])
    start = time.time()
    for exp_num in range(10):
        acc_notes[exp_num, :] = main()
        print(acc_notes[exp_num, :])
    df = pd.DataFrame(acc_notes, columns=["factory1_acc", "factory2_acc", "factory3_acc"])
    df.to_csv(f'{dataset}.csv')