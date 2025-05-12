import torch
import numpy as np
import random


# 设置随机数种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
# setup_seed(0)
#
# print(np.random.rand(3))
# print(np.random.rand(3))
# [0.35843357 0.72730695 0.10236735]
# [0.82672992 0.64361755 0.85369816]


# [4.17022005e-01 7.20324493e-01 1.14374817e-04]
# [0.30233257 0.14675589 0.09233859]
# [0.18626021 0.34556073 0.39676747]