import torch
from torch.functional import einsum
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary

def parameter_check(expand_sizes, out_channels, num_token, d_model, in_channel, project_demension, fc_demension, num_class):
    check_list = [[],[]]
    for i in range(len(expand_sizes)):
        check_list[0].extend(expand_sizes[i])
        check_list[1].extend(out_channels[i])
    for i in range(len(check_list[0]) - 1):
        assert check_list[0][i + 1] % check_list[1][i] == 0 , 'The out_channel should be divisible by expand_size of the next block, due to the expanded DW conv'
    assert num_token > 0, 'num_token should be larger than 0'
    assert d_model > 0, 'd_model should be larger than 0'
    assert in_channel > 0, 'in_channel should be larger than 0'
    assert project_demension > 0, 'project_demension should be larger than 0'
    assert fc_demension > 0, 'fc_demension should be larger than 0'
    assert num_class > 0, 'num_class should be larger than 0'


class BottleneckLite(nn.Module):
    '''Proposed in Yunsheng Li, Yinpeng Chen et al., MicroNet, arXiv preprint arXiv: 2108.05894v1'''
    def __init__(self, in_channel, expand_size, out_channel, kernel_size=3, stride=1, padding=1):
        super(BottleneckLite, self).__init__()
        self.bnecklite = nn.Sequential(
            nn.Conv2d(in_channel, expand_size, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channel),
            nn.ReLU6(inplace=True),
            nn.Conv2d(expand_size, out_channel, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channel)
        )
    
    def forward(self, x):
        return self.bnecklite(x)


class MLP(nn.Module):
    '''widths [in_channel, ..., out_channel], with ReLU within'''
    def __init__(self, widths, bn=True):
        super(MLP, self).__init__()
        layers = []
        for n in range(len(widths) - 1):
            layer_ = nn.Sequential(
                nn.Linear(widths[n], widths[n + 1]),
                nn.ReLU6(inplace=True),
            )
            layers.append(layer_)
        self.mlp = nn.Sequential(*layers)
        if bn:
            self.mlp = nn.Sequential(
                *layers,
                nn.BatchNorm1d(widths[-1])
            )

    def forward(self, x):
        return self.mlp(x)


class DynamicReLU(nn.Module):
    '''channel-width weighted DynamticReLU '''
    '''Yinpeng Chen, Xiyang Dai et al., Dynamtic ReLU, arXiv preprint axXiv: 2003.10027v2'''
    def __init__(self, in_channel, control_demension, k=2):
        super(DynamicReLU, self).__init__()
        self.in_channel = in_channel
        self.k = k
        self.control_demension = control_demension
        self.Theta = MLP([control_demension, 4 * control_demension, 2 * k * in_channel], bn=True)

    def forward(self, x, control_vector):
        n, _, _, _ = x.shape
        a_default = torch.ones(n, self.k * self.in_channel)
        a_default[:, self.k * self.in_channel // 2 : ] = torch.zeros(n, self.k * self.in_channel // 2)
        theta = self.Theta(control_vector)
        theta = 2 * torch.sigmoid(theta) - 1
        a = theta[:, 0 : self.k * self.in_channel] + a_default
        b = theta[:, self.k * self.in_channel : ] * 0.5
        a = a.reshape(n, self.k, self.in_channel)
        b = b.reshape(n, self.k, self.in_channel)
        # x (NCHW), a & b (N, k, C)
        x = einsum('nchw, nkc -> nchwk', x, a) + einsum('nchw, nkc -> nchwk', torch.ones_like(x), b)
        return x.max(4)[0]


class Mobile(nn.Module):
    '''Without shortcut, if stride=2, donwsample, DW conv expand channel, PW conv squeeze channel'''
    def __init__(self, in_channel, expand_size, out_channel, token_demension, kernel_size=3, stride=1, k=2):
        super(Mobile, self).__init__()
        self.stride = stride
        if stride == 2:
            self.strided_conv = nn.Sequential(
                nn.Conv2d(in_channel, expand_size, kernel_size=3, stride=2, padding=int(kernel_size // 2), groups=in_channel),
                nn.BatchNorm2d(expand_size),
                nn.ReLU6(inplace=True)
            )
            self.conv1 = nn.Conv2d(expand_size, in_channel, kernel_size=1, stride=1)
            self.bn1 = nn.BatchNorm2d(in_channel)
            self.ac1 = DynamicReLU(in_channel, token_demension, k=k)      
            self.conv2 = nn.Conv2d(in_channel, expand_size, kernel_size=3, stride=1, padding=1, groups=in_channel)
            self.bn2 = nn.BatchNorm2d(expand_size)
            self.ac2 = DynamicReLU(expand_size, token_demension, k=k)          
            self.conv3 = nn.Conv2d(expand_size, out_channel, kernel_size=1, stride=1)
            self.bn3 = nn.BatchNorm2d(out_channel)
        else:
            self.conv1 = nn.Conv2d(in_channel, expand_size, kernel_size=1, stride=1)
            self.bn1 = nn.BatchNorm2d(expand_size)
            self.ac1 = DynamicReLU(expand_size, token_demension, k=k)      
            self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=3, stride=1, padding=1, groups=expand_size)
            self.bn2 = nn.BatchNorm2d(expand_size)
            self.ac2 = DynamicReLU(expand_size, token_demension, k=k)          
            self.conv3 = nn.Conv2d(expand_size, out_channel, kernel_size=1, stride=1)
            self.bn3 = nn.BatchNorm2d(out_channel)

    def forward(self, x, first_token):
        if self.stride == 2:
            x = self.strided_conv(x)
        x = self.bn1(self.conv1(x))
        x = self.ac1(x, first_token)
        x = self.bn2(self.conv2(x))
        x = self.ac2(x, first_token)
        return self.bn3(self.conv3(x))


class Mobile_Former(nn.Module):
    '''Local feature -> Global feature'''
    def __init__(self, d_model, in_channel):
        super(Mobile_Former, self).__init__()
        self.project_Q = nn.Linear(d_model, in_channel)
        self.unproject = nn.Linear(in_channel, d_model)
        self.eps = 1e-10
        self.shortcut = nn.Sequential()

    def forward(self, local_feature, x):
        n, c, _, _ = local_feature.shape
        local_feature = local_feature.view(n, c, -1).permute(0, 2, 1)   # N, L, C
        project_Q = self.project_Q(x)   # N, M, C
        scores = torch.einsum('nmc , nlc -> nml', project_Q, local_feature) / (np.sqrt(c) + self.eps)
        scores_map = F.softmax(scores, dim=-1)  # each m to every l
        fushion = torch.einsum('nml, nlc -> nmc', scores_map, local_feature)
        unproject = self.unproject(fushion) # N, m, d
        return unproject + self.shortcut(x)


class Former(nn.Module):
    '''Post LayerNorm, no Res according to the paper.'''
    def __init__(self, head, d_model, expand_ratio=2):
        super(Former, self).__init__()
        self.d_model = d_model
        self.eps = 1e-10
        self.heads = head
        assert d_model % head == 0
        self.d_per_head = d_model // head
        self.QVK = MLP([d_model, d_model * 3], bn=False)
        self.Q_to_heads = MLP([d_model, d_model], bn=False)
        self.K_to_heads = MLP([d_model, d_model], bn=False)
        self.V_to_heads = MLP([d_model, d_model], bn=False)
        self.heads_to_o = MLP([d_model, d_model], bn=False)
        self.norm = nn.LayerNorm(d_model)
        self.mlp = MLP([d_model, expand_ratio * d_model, d_model], bn=False)
        self.mlp_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        n, m, d = x.shape
        QVK = self.QVK(x)
        Q = QVK[:, :, 0: self.d_model]
        Q = self.Q_to_heads(Q).reshape(n, m, self.d_model // self.heads, self.heads)   # (n, m, d/head, head)
        K = QVK[:, :, self.d_model: 2 * self.d_model]
        K = self.K_to_heads(K).reshape(n, m, self.d_model // self.heads, self.heads)   # (n, m, d/head, head)
        V = QVK[:, :, 2 * self.d_model: 3 * self.d_model]
        V = self.V_to_heads(V).reshape(n, m, self.d_model // self.heads, self.heads)   # (n, m, d/head, head)
        scores = torch.einsum('nqdh, nkdh -> nhqk', Q, K) / (np.sqrt(self.d_per_head) + self.eps)   # (n, h, q, k)
        scores_map = F.softmax(scores, dim=-1)  # (n, h, q, k)
        v_heads = torch.einsum('nkdh, nhqk -> nhqd', V, scores_map).permute(0, 2, 1, 3) #   (n, h, m, d_p) -> (n, m, h, d_p)
        v_heads = v_heads.reshape(n, m, d)
        attout = self.heads_to_o(v_heads)
        attout = self.norm(attout)  #post LN
        attout = self.mlp(attout)
        attout = self.mlp_norm(attout)  # post LN
        return attout   # No res


class Former_Mobile(nn.Module):
    '''Global feature -> Local feature'''
    def __init__(self, d_model, in_channel):
        super(Former_Mobile, self).__init__()
        self.project_KV = MLP([d_model, 2 * in_channel], bn=False)
        self.shortcut = nn.Sequential()
    
    def forward(self, x, global_feature):
        res = self.shortcut(x)
        n, c, h, w = x.shape
        project_kv = self.project_KV(global_feature)
        K = project_kv[:, :, 0 : c]  # (n, m, c)
        V = project_kv[:, :, c : ]   # (n, m, c)
        x = x.reshape(n, c, h * w).permute(0, 2, 1) # (n, l, c) , l = h * w
        scores = torch.einsum('nqc, nkc -> nqk', x, K) # (n, l, m)
        scores_map = F.softmax(scores, dim=-1) # (n, l, m)
        v_agg = torch.einsum('nqk, nkc -> nqc', scores_map, V)  # (n, l, c)
        feature = v_agg.permute(0, 2, 1).reshape(n, c, h, w)
        return feature + res

class MobileFormerBlock(nn.Module):
    '''main sub-block, input local feature (N, C, H, W) & global feature (N, M, D)'''
    '''output local & global, if stride=2, then it is a downsample Block'''
    def __init__(self, in_channel, expand_size, out_channel, d_model, stride=1, k=2, head=8, expand_ratio=2):
        super(MobileFormerBlock, self).__init__()
        self.mobile = Mobile(in_channel, expand_size, out_channel, d_model, kernel_size=3, stride=stride, k=2)
        self.former = Former(head, d_model, expand_ratio=expand_ratio)
        self.mobile_former = Mobile_Former(d_model, in_channel)
        self.former_mobile = Former_Mobile(d_model, out_channel)
    
    def forward(self, local_feature, global_feature):
        z_hidden = self.mobile_former(local_feature, global_feature)
        z_out = self.former(z_hidden)
        x_hidden = self.mobile(local_feature, z_out[:, 0, :])
        x_out = self.former_mobile(x_hidden, z_out)
        return x_out, z_out

class MobileFormer(nn.Module):
    '''Resolution should larger than [2 ** (num_stages + 1) + 7]'''
    '''stem -> bneck-lite -> stages(strided at first block) -> up-project-1x1 -> avg-pool -> fc1 -> scores-fc'''
    def __init__(self, expand_sizes=None, out_channels=None, 
                       num_token=6, d_model=192, in_channel=3, 
                       project_demension=1152, fc_demension=None, num_class=None):
        super(MobileFormer, self).__init__()

        parameter_check(expand_sizes, out_channels, num_token, d_model, in_channel, project_demension, fc_demension, num_class)
        
        self.num_token, self.d_model = num_token, d_model
        self.tokens = nn.Parameter(torch.randn(1, self.num_token, d_model), requires_grad=True)
        self.inter_channel = 16
        self.num_stages = len(expand_sizes)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channel, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True)
        )
        self.bneck = BottleneckLite(16, 32, 16, kernel_size=3, stride=1, padding=1)
        self.blocks = []
        for num_stage in range(self.num_stages):
            num_blocks = len(expand_sizes[num_stage])
            for num_block in range(num_blocks):
                if num_block == 0:
                    self.blocks.append(
                        MobileFormerBlock(self.inter_channel, expand_sizes[num_stage][num_block], out_channels[num_stage][num_block], d_model, stride=2)
                    )
                    self.inter_channel = out_channels[num_stage][num_block]
                else:
                    self.blocks.append(
                        MobileFormerBlock(self.inter_channel, expand_sizes[num_stage][num_block], out_channels[num_stage][num_block], d_model, stride=1)
                    )
                    self.inter_channel = out_channels[num_stage][num_block]

        self.project = nn.Conv2d(self.inter_channel, project_demension, kernel_size=1, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(project_demension + d_model, fc_demension),
            nn.ReLU6(inplace=True),
            nn.Linear(fc_demension, num_class)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.bneck(x)
        tokens = torch.zeros(x.shape[0], self.num_token, self.d_model)
        for i in range(x.shape[0]):
            tokens[i, :, :] = self.tokens
        for block in self.blocks:
            x, tokens = block(x, tokens)
        x = self.project(x)
        x = self.avgpool(x).squeeze()
        x = torch.cat([x, tokens[:, 0, :]], dim=-1)
        return self.fc(x)


if __name__ == '__main__':
    print()
    print('############################### Inference check (MobileFormer 294M)###############################')

    args = {
        'expand_sizes' : [[96, 96], [144, 192], [288, 384, 576, 768], [768, 1152, 1152]],
        'out_channels' : [[24, 24], [48, 48], [96, 96, 128, 128], [192, 192, 192]],
        'num_token' : 6, 'd_model' : 192, 
        'in_channel' : 3, 'project_demension' : 1152, 'fc_demension' : 1920, 'num_class' : 100
    }
    print()
    model = MobileFormer(**args)
    summary(model, (3, 224, 224))
