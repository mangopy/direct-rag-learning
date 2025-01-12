import os
from prepare import run_data_prepare, offpolicy, re_weighted
from tuning import train
import torch
import numpy as np
import random

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multiple gpus
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


if __name__ == '__main__':
    procedure = os.environ['PROCEDURE']
    if procedure == 'prepare':
        run_data_prepare()
    elif procedure == 'offpolicy':
        offpolicy()
    elif procedure == 'train':
        train()
    elif procedure == 'weight':
        re_weighted()
    else:
        raise "Not implemented procedure ..."