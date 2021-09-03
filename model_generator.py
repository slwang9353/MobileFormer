from model import MobileFormer
import torch
from torch.nn import init
import torch.nn as nn

def initNetParams(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.1)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.1)

def mobile_former_custom(args, pre_train=False, state_dir=None):
    model = MobileFormer(**args)
    if pre_train:
        model.load_state_dict(torch.load(state_dir))
    else:
        initNetParams(model)
        print('Model initialized.')
    return model

def mobile_former_508(num_class, pre_train=False, state_dir=None):
    args = {
        'expand_sizes' : [[144, 120], [240, 216], [432, 512, 768, 1056], [1056, 1440, 1440]],
        'out_channels' : [[40, 40], [72, 72], [128, 128, 176, 176], [240, 240, 240]],
        'num_token' : 6, 'd_model' : 192, 
        'in_channel' : 3, 'stem_out_channel' : 24, 
        'bneck_exp' : 48, 'bneck_out' : 24,
        'project_demension' : 1440, 'fc_demension' : 1920, 'num_class' : num_class
    }
    model = MobileFormer(**args)
    if pre_train:
        print('Modle loading...')
        model.load_state_dict(torch.load(state_dir))
        print('Modle loaded.')
    else:
        initNetParams(model)
        print('Model initialized.')
    return model

def mobile_former_294(num_class, pre_train=False, state_dir=None):
    args = {
        'expand_sizes' : [[96, 96], [144, 192], [288, 384, 576, 768], [768, 1152, 1152]],
        'out_channels' : [[24, 24], [48, 48], [96, 96, 128, 128], [192, 192, 192]],
        'num_token' : 6, 'd_model' : 192, 
        'in_channel' : 3, 'stem_out_channel' : 16, 
        'bneck_exp' : 32, 'bneck_out' : 16,
        'project_demension' : 1152, 'fc_demension' : 1920, 'num_class' : num_class
    }
    model = MobileFormer(**args)
    if pre_train:
        print('Modle loading...')
        model.load_state_dict(torch.load(state_dir))
        print('Modle loaded.')
    else:
        initNetParams(model)
        print('Model initialized.')
    return model

def mobile_former_214(num_class, pre_train=False, state_dir=None):
    args = {
        'expand_sizes' : [[72, 60], [120, 160], [240, 320, 480, 672], [672, 960, 960]],
        'out_channels' : [[20, 20], [40, 40], [80, 80, 112, 112], [160, 160, 160]],
        'num_token' : 6, 'd_model' : 192, 
        'in_channel' : 3, 'stem_out_channel' : 12, 
        'bneck_exp' : 24, 'bneck_out' : 12,
        'project_demension' : 960, 'fc_demension' : 1600, 'num_class' : num_class
    }
    model = MobileFormer(**args)
    if pre_train:
        print('Modle loading...')
        model.load_state_dict(torch.load(state_dir))
        print('Modle loaded.')
    else:
        initNetParams(model)
        print('Model initialized.')
    return model

def mobile_former_151(num_class, pre_train=False, state_dir=None):
    args = {
        'expand_sizes' : [[72, 48], [96, 96], [192, 256, 384, 528], [528, 768, 768]],
        'out_channels' : [[16, 16], [32, 32], [64, 64, 88, 88], [128, 128, 128]],
        'num_token' : 6, 'd_model' : 192, 
        'in_channel' : 3, 'stem_out_channel' : 12, 
        'bneck_exp' : 24, 'bneck_out' : 12,
        'project_demension' : 768, 'fc_demension' : 1280, 'num_class' : num_class
    }
    model = MobileFormer(**args)
    if pre_train:
        print('Modle loading...')
        model.load_state_dict(torch.load(state_dir))
        print('Modle loaded.')
    else:
        initNetParams(model)
        print('Model initialized.')
    return model

def mobile_former_96(num_class, pre_train=False, state_dir=None):
    args = {
        'expand_sizes' : [[72], [96, 96], [192, 256, 384], [528, 768]],
        'out_channels' : [[16], [32, 32], [64, 64, 88], [128, 128]],
        'num_token' : 4, 'd_model' : 128, 
        'in_channel' : 3, 'stem_out_channel' : 12, 
        'bneck_exp' : 24, 'bneck_out' : 12,
        'project_demension' : 768, 'fc_demension' : 1280, 'num_class' : num_class
    }
    model = MobileFormer(**args)
    if pre_train:
        print('Modle loading...')
        model.load_state_dict(torch.load(state_dir))
        print('Modle loaded.')
    else:
        initNetParams(model)
        print('Model initialized.')
    return model
