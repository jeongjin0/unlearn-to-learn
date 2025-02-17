import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, Subset

import os
import numpy as np


def create_dataloader(args, seed=2024):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if args.dataset == "cifar10":
        transform_train = create_transforms(args.dataset, is_train=True)
        transform_test = create_transforms(args.dataset, is_train=False)
        
        full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        
        train_size = 45000
        val_size = 5000
        trainset, valset = random_split(full_trainset, [train_size, val_size], 
                                        generator=torch.Generator().manual_seed(seed))
        
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    elif args.dataset == "timagenet":
        data_dir = "data/tiny-imagenet-200/"
        transform_train = create_transforms(args.dataset, is_train=True)
        transform_test = create_transforms(args.dataset, is_train=False)
        
        trainset = torchvision.datasets.ImageFolder(os.path.join(data_dir, "train"), transform_train)
        valset = torchvision.datasets.ImageFolder(os.path.join(data_dir, "val"), transform_test)
        testset = torchvision.datasets.ImageFolder(os.path.join(data_dir, "test"), transform_test)
        
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    return train_loader, val_loader, test_loader


def create_transforms(dataset, is_train):
    if dataset == "cifar10":
        if is_train:
            return transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(degrees=15),
                    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        else:
            return transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        
    elif dataset == "timagenet":
        if is_train:
            return transforms.Compose([
                    transforms.RandomResizedCrop(64),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(degrees=20),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        else:
            return transforms.Compose([
                    transforms.Resize(64),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])