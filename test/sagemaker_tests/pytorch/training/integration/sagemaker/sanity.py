import torch
import numpy as np
import time
import os
import sys
import argparse

DTYPE = torch.float16
DTSIZE = 2
device = torch.device("cuda", 0)
size = 8
num_tensors=512
tests = [
    torch.ones(int(size / DTSIZE), dtype=DTYPE, device=device)
    for _ in range(num_tensors)
]
