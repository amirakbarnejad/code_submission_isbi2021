
import numpy as np
import sys
import os
import copy
import pickle
import re
from abc import ABC, abstractmethod
import math
import copy
import gc
from copy import deepcopy
from pathlib import Path
import torchvision.models as torchmodels
import openslide
import pyvips
import torch.utils.data
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



class ImgnetDenormalize(object):
    def __init__(self):
        self.tfm1 = torchvision.transforms.Normalize(mean=[0.0, 0.0, 0.0],\
                                                     std=[1.0/0.229, 1.0/0.224, 1.0/0.225])
        self.tfm2 = torchvision.transforms.Normalize(mean=[-0.485, -0.456, -0.406],\
                                                     std=[1.0, 1.0, 1.0])
    def __call__(self, x):
        if(len(list(x.size()))==3):
            #[CxHxW] case
            toret = self.tfm2(self.tfm1(x))
        else:
            #[N x C x H x W] case
            N = list(x.size())[0]
            toret = [self.tfm2(self.tfm1(x[n,:,:,:])) for n in range(N)]
            return torch.stack(toret)
        return toret
