import math
import os
import random
import shutil
import numpy as np
import torch
import sys

sys.path.append('../..')
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=1.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
                      float(max(1, num_training_steps - num_warmup_steps))  # 行尾\表示续行符
        return max(0., 0.5 * (1. + math.cos(math.pi * num_cycles * no_progress)))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)

