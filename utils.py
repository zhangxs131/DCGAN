import random
import torch
import numpy as np

def set_seed(manualSeed=999):

    print('Random Seed',manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)

