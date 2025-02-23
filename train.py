import argparse
import logging
import sys
import time
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from adv_attack import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='AT-stability')
# Hyper-parameters and neural network settings
parser.add_argument('--dataset', type=str,   default='cifar10', choices=['cifar10', 'svhn', 'cifar100'])
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--epochs', default=201, type=int)
parser.add_argument('--model', type=str,   default='PreRes18_standard')
parser.add_argument('--lr_max', default=0.1, type=float) # 0.01 for svhn
parser.add_argument('--width_factor', default=10, type=int)
parser.add_argument('--lr_schedule', default='piecewise', choices=['piecewise', 'cosine'])
parser.add_argument('--lr_chechpoint1', type=int, default=100)
parser.add_argument('--lr_chechpoint2', type=int, default=150)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=5e-4) 
# Settings for the PGD attack
parser.add_argument('--epsilon', default=8/255., type=float) 
parser.add_argument('--attack_iters', default=10, type=int) 
parser.add_argument('--pgd_alpha', default=2., type=float) # PGD step size
# Different gradient operators
parser.add_argument('--att_method', type=str, default='tanh', choices=['sign', 'tanh', 'raw', 'norm_steep']) # respectively refers to [sign-PGD, tanh_{\gamma}-PGD, RG-PGD, G_{p}-PGD]
parser.add_argument('--eval_method', type=str, default='sign', choices=['sign', 'tanh', 'raw', 'norm_steep'])
parser.add_argument('--beta', type=float,   default=1) # equivalent to tuning the \gamma value in the tanh_{\gamma} function, or p value in G_{p}-PGD.

# Other settings
parser.add_argument('--seed', default=2, type=int)  # random seed
parser.add_argument('--resume', default=0, type=int)
parser.add_argument('--chkpt_iters', default=50, type=int) # saving model parameters at some iterations
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)



#########################
pgd_alpha = args.pgd_alpha/255.

if args.dataset == 'cifar10':
    fname = 'cifar10_results'
    model_dir = 'cifar10_saved_models'
    num_class = 10

    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),])
    
    trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False, download=True, transform=transforms.ToTensor()) 
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    
elif args.dataset == 'cifar100':
    fname = 'cifar100_results'
    model_dir = 'cifar100_saved_models'
    num_class = 100

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),])

    trainset = torchvision.datasets.CIFAR100(root='./cifar100', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR100(root='./cifar100', train=False, download=True, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

elif args.dataset == 'svhn':
    fname = 'svhn_results'
    model_dir = 'svhn_saved_models'
    num_class = 10

    transform_train = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.ToTensor(),])

    trainset = torchvision.datasets.SVHN(root='./svhn', split='train', transform=transform_train, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.SVHN(root='./svhn', split='test', transform=transforms.ToTensor(), download=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

else:
    raise ValueError

if not os.path.exists(fname):
        os.makedirs(fname)

if not os.path.exists(model_dir):
        os.makedirs(model_dir)

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG,
    handlers=[
        logging.FileHandler(os.path.join(fname, f'{args.model}{args.seed}_PGD-{args.att_method}_beta{args.beta}.log')),
        logging.StreamHandler()
    ])

logger.info(args)



#### Loading model ####
if args.model == 'WideResNet':
    from models.wideresnet import *
    model = WideResNet(depth=34, num_classes=num_class, widen_factor=args.width_factor, dropRate=0.0)
elif args.model ==  'PreRes18_standard':
    from models.Preact_ResNet_standard import *
    model = PreActResNet18(num_classes=num_class)
else:
    raise ValueError


model = model.to(device)
if device == 'cuda':
    model = nn.DataParallel(model)
    cudnn.benchmark = True
opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)
criterion = nn.CrossEntropyLoss()
if args.lr_schedule == 'piecewise':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, [args.lr_chechpoint1, args.lr_chechpoint2], 0.1)
elif args.lr_schedule == 'cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)


#### Train and test ####
def train():
    model.train()   

    train_loss = 0
    train_acc = 0

    train_robust_loss = 0
    train_robust_acc = 0

    train_sgnPGD_loss = 0
    train_sgnPGD_acc = 0

    train_n = 0
    
    for batch_id, (X, y) in enumerate(trainloader):
        X, y = X.to(device), y.to(device)
        train_n += y.size(0)
        
        # Evaluate the model on the clean data 
        with torch.no_grad():
            output = model(X)
            loss = criterion(output, y)
            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()

        # Evaluate the model with a specific PGD variant 
        X_adv = pgd_attack(model, X, y, args.epsilon, args.attack_iters, 
                            pgd_alpha, att_method = args.eval_method, beta=args.beta)
        with torch.no_grad():
            sign_PGD_output = model(X_adv)
            sign_PGD_loss = criterion(sign_PGD_output, y)
            train_sgnPGD_loss += sign_PGD_loss.item() * y.size(0)
            train_sgnPGD_acc += (sign_PGD_output.max(1)[1] == y).sum().item()
            
        # Train the model using a specific PGD variant
        X_adv = pgd_attack(model, X, y, args.epsilon, args.attack_iters, 
                            pgd_alpha, att_method = args.att_method, beta = args.beta)
        robust_output = model(X_adv)
        robust_loss = criterion(robust_output, y)
        assert not torch.isinf(robust_loss).any()
        assert not torch.isnan(robust_loss).any()
        opt.zero_grad()
        robust_loss.backward()
        # Update model parameters   
        opt.step()
        
        train_robust_loss += robust_loss.item() * y.size(0)
        train_robust_acc += (robust_output.max(1)[1] == y).sum().item()    
      
    return train_loss/train_n, train_acc/train_n, train_robust_loss/train_n, train_robust_acc/train_n, train_sgnPGD_loss/train_n, train_sgnPGD_acc/train_n


def evaluate(mode, epsilon, att_iters, pgd_alpha):
    model.eval()
    test_loss = 0
    test_acc = 0
    test_n = 0
    for _, (X, y) in enumerate(testloader):
        X, y = X.to(device), y.to(device)
        if mode == 'clean':
            X_adv = X
        else:
            X_adv = pgd_attack(model, X, y, args.epsilon, args.attack_iters, 
                              pgd_alpha, mode, args.beta)
   
        with torch.no_grad():
            output = model(X_adv)
            loss = criterion(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            test_n += y.size(0)
    return test_loss/test_n, test_acc/test_n


#########
if args.resume: # Resume training from a certain checkpoint
    start_epoch = args.resume
    model.load_state_dict(torch.load(os.path.join(model_dir, f'{args.model}{args.seed}_epoch{args.resume}_PGD-{args.att_method}_beta{args.beta}.pth')))
    opt.load_state_dict(torch.load(os.path.join(model_dir, f'opt_{args.model}{args.seed}_epoch{args.resume}_PGD-{args.att_method}_beta{args.beta}.pth')))
    logger.info(f'Resuming at epoch {start_epoch}')
else:
    start_epoch = 0

logger.info('Epoch \t Train Time \t Test Time \t LR \t \t Train Loss \t Train Acc \t Train Adv Loss \t Train Adv Acc \t Train Surrogate Loss \t Train Surrogate Acc \t Test Loss \t Test Acc \t Test Adv Loss \t Test Adv Acc \t Test Surrogate Loss \t Test Surrogate Acc ')
for epoch in range(start_epoch, args.epochs):
    
    start_time = time.time()
    train_loss, train_acc, train_robust_loss, train_robust_acc, train_sgnPGD_loss, train_sgnPGD_acc  = train()
    train_time = time.time()
    scheduler.step()  # upgrade the learning rate after each epoch

    # Evaluate model on the test set using three different J-losses
    test_loss, test_acc = evaluate(mode = 'clean', epsilon = 0, 
                                                            att_iters=0, 
                                                            pgd_alpha=0)

    test_robust_loss, test_robust_acc = evaluate(mode = args.att_method, epsilon = args.epsilon,  
                                                                        att_iters=args.attack_iters, 
                                                                        pgd_alpha=pgd_alpha)

    test_sgnPGD_loss, test_sgnPGD_acc = evaluate(mode = args.eval_method, epsilon = args.epsilon,  
                                                                        att_iters=args.attack_iters, 
                                                                        pgd_alpha=pgd_alpha)
    test_time = time.time()
        

    logger.info('%d \t\t %.1f \t \t %.1f \t \t %.4f  \t\t %.4f \t %.4f \t \t %.4f  \t %.4f  \t \t %.4f \t %.4f  \t \t %.4f \t %.4f  \t \t %.4f \t %.4f  \t \t %.4f \t %.4f',
        epoch, train_time - start_time, test_time - train_time, scheduler.get_last_lr()[0],
        train_loss, train_acc, train_robust_loss, train_robust_acc, train_sgnPGD_loss, train_sgnPGD_acc,
        test_loss, test_acc, test_robust_loss, test_robust_acc, test_sgnPGD_loss, test_sgnPGD_acc)


    #save checkpoint
    if (epoch+1) % args.chkpt_iters == 0 or epoch == (args.epochs-1):
        torch.save(model.state_dict(), os.path.join(model_dir, f'{args.model}{args.seed}_epoch{epoch}_PGD-{args.att_method}_beta{args.beta}.pth'))
        torch.save(opt.state_dict(), os.path.join(model_dir, f'opt_{args.model}{args.seed}_epoch{epoch}_PGD-{args.att_method}_beta{args.beta}.pth'))

