import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from collections import defaultdict
import random

def get_model(args, device, model="resnet18"):
    if args.dataset == "cifar10":
        num_classes = 10
        imgs = 32
        patch_size = 4
    elif args.dataset == "timagenet":
        num_classes = 200
        imgs = 64
        patch_size = 8

    if model == "resnet18":
        from models.resnet import resnet18
        model = resnet18(num_classes=num_classes)

    elif model == "vgg16bn":
        from models.vgg import vgg16_bn
        model = vgg16_bn(num_classes=num_classes, imgs=imgs)

    elif model == "vgg16":
        from models.vgg import vgg16
        model = vgg16(num_classes=num_classes, imgs=imgs)

    elif model == "vit":
        from models.vit import ViT
        model = ViT(
            image_size = imgs,
            patch_size = patch_size,
            num_classes = num_classes,
            dim = 512,
            depth = 6,
            heads = 8,
            mlp_dim = 512,
            dropout = 0.1,
            emb_dropout = 0.1)
    
    elif model == "vit_timm":
        import timm
        model = timm.create_model("vit_base_patch16_384", pretrained=True)
        model.head = nn.Linear(model.head.in_features, num_classes)

    elif model == "cait":
        from models.cait import CaiT
        model = CaiT(
            image_size = 32,
            patch_size = 4,
            num_classes = 10,
            dim = 512,
            depth = 6,   # depth of transformer for patch to patch attention only
            cls_depth=2, # depth of cross attention of CLS tokens to patch
            heads = 8,
            mlp_dim = 512,
            dropout = 0.1,
            emb_dropout = 0.1,
            layer_dropout = 0.05)

    if args.load_path != None:
        state_dict = torch.load(args.load_path)
        model.load_state_dict(state_dict)

    #-----------For fine-pruning defense
    #new_state_dict = {}
    #for key, value in state_dict.items():
    #    if key == "linear.weight_orig":
    #        new_state_dict["linear.weight"] = value * state_dict['linear.weight_mask']
    #    elif key != "linear.weight_mask":
    #        new_state_dict[key] = value

    # Load the modified state dictionary into the model
   # model.load_state_dict(new_state_dict)

    model.to(device)

    if device == 'cuda':
        model = torch.nn.DataParallel(model, device_ides=[0,1])
        cudnn.benchmark = True

    return model
