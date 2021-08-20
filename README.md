# MobileFormer
MobileFormer in torch
# Including
    Mobile-Former proposed in: Yinpeng Chen, Xiyang Dai et al., Mobile-Former: Bridging MobileNet and Transformer. arxiv.org/abs/2108.05895
    Dynamtic ReLU proposed in: Yinpeng Chen, Xiyang Dai et al., Dynamtic ReLU. arxiv.org/abs/2003.10027v2
    Lite-BottleNeck proposed in: Yunsheng Li, Yinpeng Chen et al., MicroNet: Improving Image Recognition with Extremely Low FLOPs. arxiv.org/abs/2108.05894v1
# Note
    The out_channel should be divisible by expand_size of the next block, due to the expanded DW conv used in strided Mobile-Former blocks.
Only the model for now.
