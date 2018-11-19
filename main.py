import argparse
import os
import random
import time
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from densenet import densenet121
from resnet import resnet18
from utils import save_checkpoint, AverageMeter, adjust_learning_rate, accuracy
from train import train, validate

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', default='~/data/nih_chest', metavar='DIR', help='path to dataset')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=120, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--epoch_decay', default=30, type=int, metavar='N', help='adjust learning rate per N epochs')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', default=256, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--weight_decay', default=0, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print_freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate', default=False,  dest='evaluate', action='store_true', help='evaluate model')
parser.add_argument('--pretrained', default=False, dest='pretrained', action='store_true', help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
parser.add_argument('--gpu', default=True, dest='gpu', action='store_true', help='enable GPU.')
parser.add_argument('--resnet', default=False, dest='resnet', action='store_true', help='train with resnet18')
parser.add_argument('--logging',default= True, help ='logging')

best_acc1 = 0


def main():
    global args, best_acc1
    args = parser.parse_args()

    if args.logging:
        filename = 'resnet' if args.resnet else 'densenet'
        filename += 'pretrained' if args.pretrained else ''
        f = open(filename+'.txt','w')


    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # create model
    if args.resnet:
        print("=> creating resnet model")
        model = resnet18(pretrained=args.pretrained)
        f.write("Model: resnet \n")
    else:
        print("=> creating densenet model")
        model = densenet121(pretrained=args.pretrained)
        f.write("Model: densenet \n")

    if args.gpu:
        model = model.cuda()
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.999))

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    f.write("File_directory: "+args.data+"\n")
    args.data = os.path.expanduser(args.data)
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(traindir,
                                         transforms.Compose([transforms.Resize(256),
                                                             transforms.CenterCrop(224),
                                                             transforms.RandomHorizontalFlip(),
                                                             transforms.ToTensor(),
                                                             normalize, ]))

    val_dataset = datasets.ImageFolder(valdir,
                                       transforms.Compose([transforms.Resize(256),
                                                           transforms.CenterCrop(224),
                                                           transforms.ToTensor(),
                                                           normalize, ]))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args.gpu, args.print_freq, f)
        return

    mode = 'evaluate' if args.evaluate else 'training'
    f.write("Mode :" + mode + "\n")
    
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, args.lr, epoch, args.epoch_decay)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args.gpu, args.print_freq, f)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args.gpu, args.print_freq,f)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best, model = 'resnet18' if args.resnet else 'densenet121')

    f.write("Best accuracy :" + str(best_acc1))
    f.close()


if __name__ == '__main__':
    main()
