# torch_seed.py
import os
import random
import numpy as np
import torch
# Set random seed for reproducibility
def torch_seed(seed: int = 123) -> None:
    # Set environment variables
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # warn-only is for AdaptiveAvgPool2d which has no deterministic implementation.
    torch.use_deterministic_algorithms(True, warn_only=False) # warn_onlyがTrueだと再現性が出ないが、Falseだと警告が出る
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed) #note: いらないかも
    random.seed(seed) # note: いらないかも