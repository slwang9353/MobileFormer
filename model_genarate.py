from model import *


def mobile_former_custom(args, pre_train=True, state_dir=None):
    model = MobileFormer(**args)
    if pre_train:
        model.load_state_dict(torch.load(state_dir))
    return model

def mobile_former_294(num_class, pre_train=True, state_dir=None):
    args = {
        'expand_sizes' : [[96, 96], [144, 192], [288, 384, 576, 768], [768, 1152, 1152]],
        'out_channels' : [[24, 24], [48, 48], [96, 96, 128, 128], [192, 192, 192]],
        'num_token' : 6, 'd_model' : 192, 
        'in_channel' : 3, 'project_demension' : 1152, 'fc_demension' : 1920, 'num_class' : num_class
    }
    model = MobileFormer(**args)
    if pre_train:
        model.load_state_dict(torch.load(state_dir))
    return model