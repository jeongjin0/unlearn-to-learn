import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import argparse
import os

from train.clean_train import train
from train.test import test
from dataloader import create_dataloader
from utils import get_model


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--test_num', type=int, default=100)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--optimizer', type=str, default="sgd", help='sgd or adam')
parser.add_argument('--model', type=str, default="resnet18")
parser.add_argument('--save_path', type=str, default="checkpoints")
parser.add_argument('--load_path', type=str, default=None)
parser.add_argument('--dataset', type=str, default="cifar10", help='cifar10 or timagenet')
parser.add_argument('--device_num', type=int, default=0)
args = parser.parse_args()

filename = f"/stage1_{args.model}_{str(args.num_epochs)}.pt"
save_path = f"{args.save_path}/{args.dataset}/{args.model}/{filename}"

print("\n--------Parameters--------")
print("batch_size:", args.batch_size)
print("num_workers:", args.num_workers)
print("momentum:", args.momentum)
print("weight_decay:", args.weight_decay)
print("test_num:", args.test_num)
print("num_epochs:", args.num_epochs)
print("lr:", args.lr)
print("load_path:", args.load_path)
print("save_path:", args.save_path + filename)
print("dataset:", args.dataset)
print("model:", args.model)
print("optimizer:", args.optimizer)
print()

trainloader, valloader, testloader, = create_dataloader(args)

device = torch.device(f"cuda:{args.device_num}" if torch.cuda.is_available() else "cpu")
model = get_model(args, device, model=args.model)
criterion = nn.CrossEntropyLoss()

if args.optimizer == "adam":
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
elif args.optimizer == "sgd":
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

acc, asr = test(model=model, dataloader=testloader, device=device, test_num=args.test_num)
print(f"(test) Acc {acc} ASR {asr}\n")

for epoch in range(args.num_epochs):
    running_loss = train(
        model=model,
        trainloader=trainloader,
        testloader=testloader,
        optimizer=optimizer,
        device=device,
        criterion=criterion,
        epoch=epoch,
        test_num=args.test_num)

    acc, asr = test(model, testloader, device, args.test_num)
    acc_train, asr_train = test(model, trainloader, device, args.test_num)
    print('[Epoch %2d Finished] Acc: %.3f Acc_Train %.3f Asr: %.3f loss: %.3f Lr: %.8f' % (epoch + 1, acc, acc_train, asr, running_loss, scheduler.get_last_lr()[0]))

    scheduler.step()

print('Finished Training')
torch.save(model.state_dict(), save_path)
print("model saved at: ", save_path)