import os, time, gc, json, pickle, argparse, math
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
from data.util import *


def num_params(model):
    return sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])


def switch_schedule(schedule, mult, switch):
    """ Apply LR multiplier before iteration "switch" """

    def f(e):
        s = schedule(e)
        if e < switch:
            return s * mult
        return s

    return f


def linear_schedule(args):
    def f(e):
        if e <= args.warmup:
            return e / args.warmup
        return max((e - args.iterations) / (args.warmup - args.iterations), 0)

    return f
