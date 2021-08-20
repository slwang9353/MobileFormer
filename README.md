# MobileFormer
MobileFormer in torch
# Including
    [1] Mobile-Former proposed in: 
                            Yinpeng Chen, Xiyang Dai et al., Mobile-Former: Bridging MobileNet and Transformer. 
                            arxiv.org/abs/2108.05895
    [2] Dynamtic ReLU proposed in: 
                            Yinpeng Chen, Xiyang Dai et al., Dynamtic ReLU. 
                            arxiv.org/abs/2003.10027v2
    [3] Lite-BottleNeck proposed in: 
                            Yunsheng Li, Yinpeng Chen et al., MicroNet: Improving Image Recognition with Extremely Low FLOPs. 
                            arxiv.org/abs/2108.05894v1
    [4] Adam-W proposed in :
                            Ilya Loshchilov & Frank Hutter, Decoupled Weight Decay Regularization.
                            arxiv.org/abs/1711.05101v3                      
# Note
    (1) Due to the expanded DW conv used in strided Mobile-Former blocks, 
        the out_channel should be divisible by expand_size of the next block.
    (2) Adam-W is embedded in train.py.
    (3) Use run() in train.py to train or search. There is an example in the train.py.
# No pre-train parameters for now.
