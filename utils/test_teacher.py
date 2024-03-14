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
from models import cnn13_stu as models_stu
from torchvision import utils
from models import attention
from models import generator as gen
import numpy

def test_teacher1(model,teacher_model):
    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])
    data_test_1 = polaroid.Polaroid('./data/' + "factory1", train=False, transform=transform_test)
    data_test_loader_1 = DataLoader(data_test_1, batch_size=50, num_workers=8)
    data_test_2 = polaroid.Polaroid('./data/' + "factory2", train=False, transform=transform_test)
    data_test_loader_2 = DataLoader(data_test_2, batch_size=50, num_workers=8)
    data_test_3 = polaroid.Polaroid('./data/' + "factory3", train=False, transform=transform_test)
    data_test_loader_3 = DataLoader(data_test_3, batch_size=50, num_workers=8)
    device = torch.device('cuda', 0)
    with torch.no_grad():
        total_correct1 = 0
        for k, (images, labels) in enumerate(data_test_loader_1):
            images = images.to(device)
            labels = labels.to(device)
            model.eval()
            output, feature = teacher_model(images, out_feature=True)
            f11 = feature[0]
            f22 = feature[1]
            f33 = feature[2]
            output = model(images,f11=f11,f22=f22,f33=f33,out_feature=False, teaching1=True,teaching2=False,teaching3=False)
            pred = output.data.max(1)[1]
            # print(pred.shape)
            total_correct1 += pred.eq(labels.data.view_as(pred)).sum()
            total_correct11=float(total_correct1) / len(data_test_1)
        total_correct2 = 0
        for k, (images, labels) in enumerate(data_test_loader_2):
            images = images.to(device)
            labels = labels.to(device)
            model.eval()
            output, feature = teacher_model(images, out_feature=True)
            f11 = feature[0]
            f22 = feature[1]
            f33 = feature[2]
            output = model(images,f11=f11,f22=f22,f33=f33,out_feature=False, teaching1=True,teaching2=False,teaching3=False)
            pred = output.data.max(1)[1]
            total_correct2 += pred.eq(labels.data.view_as(pred)).sum()
            total_correct22 = float(total_correct2) / len(data_test_2)
        total_correct3 = 0
        for k, (images, labels) in enumerate(data_test_loader_3):
            images = images.to(device)
            labels = labels.to(device)
            model.eval()
            output, feature = teacher_model(images, out_feature=True)
            f11 = feature[0]
            f22 = feature[1]
            f33 = feature[2]
            output = model(images,f11=f11,f22=f22,f33=f33,out_feature=False, teaching1=True,teaching2=False,teaching3=False)
            pred = output.data.max(1)[1]
            total_correct3 += pred.eq(labels.data.view_as(pred)).sum()
            total_correct33 = float(total_correct3) / len(data_test_3)
    accuracy = (total_correct11+total_correct22+total_correct33)/3
    return accuracy

def test_teacher2(model,teacher_model,best_teacher_model1):
    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])
    data_test_1 = polaroid.Polaroid('./data/' + "factory1", train=False, transform=transform_test)
    data_test_loader_1 = DataLoader(data_test_1, batch_size=50, num_workers=8)
    data_test_2 = polaroid.Polaroid('./data/' + "factory2", train=False, transform=transform_test)
    data_test_loader_2 = DataLoader(data_test_2, batch_size=50, num_workers=8)
    data_test_3 = polaroid.Polaroid('./data/' + "factory3", train=False, transform=transform_test)
    data_test_loader_3 = DataLoader(data_test_3, batch_size=50, num_workers=8)
    device = torch.device('cuda', 0)
    with torch.no_grad():
        total_correct1 = 0
        for k, (images, labels) in enumerate(data_test_loader_1):
            images = images.to(device)
            labels = labels.to(device)
            model.eval()
            output, feature = teacher_model(images, out_feature=True)
            output1, feature1 = best_teacher_model1(images, out_feature=True)
            f11 = feature1[0]
            f22 = feature[1]
            f33 = feature[2]
            output = model(images,f11=f11,f22=f22,f33=f33,out_feature=False, teaching1=True,teaching2=True,teaching3=False)
            pred = output.data.max(1)[1]
            # print(pred.shape)
            total_correct1 += pred.eq(labels.data.view_as(pred)).sum()
            total_correct11=float(total_correct1) / len(data_test_1)
        total_correct2 = 0
        for k, (images, labels) in enumerate(data_test_loader_2):
            images = images.to(device)
            labels = labels.to(device)
            model.eval()
            output, feature = teacher_model(images, out_feature=True)
            output1, feature1 = best_teacher_model1(images, out_feature=True)
            f11 = feature1[0]
            f22 = feature[1]
            f33 = feature[2]
            output = model(images,f11=f11,f22=f22,f33=f33,out_feature=False, teaching1=True,teaching2=True,teaching3=False)
            pred = output.data.max(1)[1]
            total_correct2 += pred.eq(labels.data.view_as(pred)).sum()
            total_correct22 = float(total_correct2) / len(data_test_2)
        total_correct3 = 0
        for k, (images, labels) in enumerate(data_test_loader_3):
            images = images.to(device)
            labels = labels.to(device)
            model.eval()
            output, feature = teacher_model(images, out_feature=True)
            output1, feature1 = best_teacher_model1(images, out_feature=True)
            f11 = feature1[0]
            f22 = feature[1]
            f33 = feature[2]
            output = model(images,f11=f11,f22=f22,f33=f33,out_feature=False, teaching1=True,teaching2=True,teaching3=False)
            pred = output.data.max(1)[1]
            total_correct3 += pred.eq(labels.data.view_as(pred)).sum()
            total_correct33 = float(total_correct3) / len(data_test_3)
    accuracy = (total_correct11+total_correct22+total_correct33)/3
    return accuracy

def test_teacher3(model,teacher_model,best_teacher_model1,best_teacher_model2):
    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])
    data_test_1 = polaroid.Polaroid('./data/' + "factory1", train=False, transform=transform_test)
    data_test_loader_1 = DataLoader(data_test_1, batch_size=50, num_workers=8)
    data_test_2 = polaroid.Polaroid('./data/' + "factory2", train=False, transform=transform_test)
    data_test_loader_2 = DataLoader(data_test_2, batch_size=50, num_workers=8)
    data_test_3 = polaroid.Polaroid('./data/' + "factory3", train=False, transform=transform_test)
    data_test_loader_3 = DataLoader(data_test_3, batch_size=50, num_workers=8)
    device = torch.device('cuda', 0)
    with torch.no_grad():
        total_correct1 = 0
        for k, (images, labels) in enumerate(data_test_loader_1):
            images = images.to(device)
            labels = labels.to(device)
            model.eval()
            output, feature = teacher_model(images, out_feature=True)
            output1, feature1 = best_teacher_model1(images, out_feature=True)
            output2, feature2 = best_teacher_model2(images, out_feature=True)
            f11 = feature1[0]
            f22 = feature2[1]
            f33 = feature[2]
            output = model(images,f11=f11,f22=f22,f33=f33,out_feature=False, teaching1=True,teaching2=True,teaching3=True)
            pred = output.data.max(1)[1]
            # print(pred.shape)
            total_correct1 += pred.eq(labels.data.view_as(pred)).sum()
            total_correct11=float(total_correct1) / len(data_test_1)
        total_correct2 = 0
        for k, (images, labels) in enumerate(data_test_loader_2):
            images = images.to(device)
            labels = labels.to(device)
            model.eval()
            output, feature = teacher_model(images, out_feature=True)
            output1, feature1 = best_teacher_model1(images, out_feature=True)
            output2, feature2 = best_teacher_model2(images, out_feature=True)
            f11 = feature1[0]
            f22 = feature2[1]
            f33 = feature[2]
            output = model(images,f11=f11,f22=f22,f33=f33,out_feature=False, teaching1=True,teaching2=True,teaching3=True)
            pred = output.data.max(1)[1]
            total_correct2 += pred.eq(labels.data.view_as(pred)).sum()
            total_correct22 = float(total_correct2) / len(data_test_2)
        total_correct3 = 0
        for k, (images, labels) in enumerate(data_test_loader_3):
            images = images.to(device)
            labels = labels.to(device)
            model.eval()
            output, feature = teacher_model(images, out_feature=True)
            output1, feature1 = best_teacher_model1(images, out_feature=True)
            output2, feature2 = best_teacher_model2(images, out_feature=True)
            f11 = feature1[0]
            f22 = feature2[1]
            f33 = feature[2]
            output = model(images,f11=f11,f22=f22,f33=f33,out_feature=False, teaching1=True,teaching2=True,teaching3=True)
            pred = output.data.max(1)[1]
            total_correct3 += pred.eq(labels.data.view_as(pred)).sum()
            total_correct33 = float(total_correct3) / len(data_test_3)
    accuracy = (total_correct11+total_correct22+total_correct33)/3
    return accuracy