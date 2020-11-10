
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



class ModuleIdentity(nn.Module):
    '''
    Dummy Identity module.
    '''
    def __init__(self):
        super(ModuleIdentity, self).__init__()

    def forward(self, x):
        return x

class ResidualEncoder(nn.Module):
    '''
    This class implements the convolutional part (i.e., without heads) of resnet.
    '''
    def __init__(self, resnettype, pretrained):
        super(ResidualEncoder, self).__init__()
        #grab privates =========
        self.resnettype = resnettype
        self.pretrained = pretrained
        #make conv_model ======
        model = resnettype(pretrained = self.pretrained)
        list_modules = list(model.children())[0:-2]
        #list_modules[-1][-1].relu = ModuleIdentity()
        self.model = nn.Sequential(*list_modules)
    
    def forward(self, x):
        return self.model(x)


class _BasicUpBlock(nn.Module):
    def __init__(self, list_channel_inouts, flag_upsample, flag_atend_applyrelu):
        super(_BasicUpBlock, self).__init__()
        #grab privates ======
        self.list_channel_inouts = list_channel_inouts
        self.flag_upsample = flag_upsample
        self.flag_atend_applyrelu = flag_atend_applyrelu
        #make internal modules ======
        self.final_relu = nn.ReLU()
        count = 0
        list_conv2d, list_bn2d, list_relu = [], [], []
        list_sequpsample = []
        for cinout in list_channel_inouts:
            if(count == 0):
                if(flag_upsample == True):
                    stride = 2
                else:
                    stride = 1
            else:
                stride = 1
            count += 1
            cin, cout = cinout
            list_sequpsample.append(nn.ConvTranspose2d(cin, cout, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False))
            list_sequpsample.append(nn.BatchNorm2d(cout, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
            if(count != len(list_channel_inouts)):
                list_sequpsample.append(nn.ReLU())
        #self.list_conv2d, self.list_bn2d, self.list_relu = list_conv2d, list_bn2d, list_relu
        self.sequpsample = nn.Sequential(*list_sequpsample)
        if(flag_upsample == True):
            self.upsample = nn.Sequential(
                    nn.ConvTranspose2d(list_channel_inouts[0][0], list_channel_inouts[-1][1],\
                              kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),\
                    nn.BatchNorm2d(list_channel_inouts[-1][1], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
        
    def forward(self, x):
        #handle identity
        identity = x + 0.0
        if(self.flag_upsample == True):
            identity = self.upsample(identity)
        #build output ====
        output = self.sequpsample(x)
        #add identity and return
        if(self.flag_atend_applyrelu == True):
            return self.final_relu(output + identity)
        else:
            return output + identity



class ResidualDecoder(nn.Module):
    '''
    This class a residual decoder (with upsampling).
    The input is a tensor of shape [N x dim_latentvector]
    The output is a tensor of shape [N x num_outputchannels x width_output x width_output]
    '''
    def __init__(self, input_view, width_output, num_outputchannels,\
                 list_upconvargs):
        '''
        Inputs:
            - input_view: a list or tuple like (64,4,4).
                          Input is originally [N x C x 1 x 1]. It has to be converted to, e.g., a [N x 64 x 4 x 4] tensor.
            - list_upconvargs: list of arguments _BasicUpBlock, a list like [[ [[128, 64],[64,64]]   , True],
                                                                             [ [[64, 32],[32, 32]]  , True],
                                                                             [ [[32, 16],[16,16]] , True]]
        '''
        super(ResidualDecoder, self).__init__()
        #grab privates ===========
        self.input_view = input_view
        self.width_output = width_output
        self.num_outputchannels = num_outputchannels 
        self.list_upconvargs = list_upconvargs
        #make basic blocks =========
        list_upbasicblock = []
        count = 0
        for uparg in self.list_upconvargs:
            new_upblock = _BasicUpBlock(uparg[0], uparg[1],\
                                        True) #the last block must not apply relu
            list_upbasicblock.append(new_upblock)
            count += 1
        self.seq_upblocks = nn.Sequential(*list_upbasicblock)
        #make conv for setting num_outputchannels =====
        self.layer_setnumoutputchannels =\
                        nn.Sequential(
                                    nn.Conv2d(self.list_upconvargs[-1][0][-1][1], self.num_outputchannels,\
                                              kernel_size=(1, 1), stride=(1, 1), bias=True)
                            )
        #make final upsample =====
        self.final_upsample = nn.Upsample(size=width_output)
    
    def forward(self, x):
        #x is [N x C x 1 x 1]
        x = x.view(-1, self.input_view[0], self.input_view[1], self.input_view[2])
        output = self.seq_upblocks(x)
        # ~ print("      before final upsample, output.shape = {}".format(output.shape))
        output = self.layer_setnumoutputchannels(output)
        output = self.final_upsample(output)
        return output
    

class ResnetVAE(nn.Module):
    def __init__(self, mod_encoder, mod_decoder, input_width, input_channels, dim_latentvector):
        super(ResnetVAE, self).__init__()
        #grab privates =====
        self.mod_encoder = mod_encoder
        self.mod_decoder = mod_decoder
        self.input_width = input_width
        self.input_channels = input_channels
        self.dim_latentvector = dim_latentvector
        #infer the size of encoder output
        self.size_encoderoutput = self._infer_sizeof_encoderoutput() #[C x H x W]
        encoder_cout = self.size_encoderoutput[0]
        #TODO: average pooling to have fewer parameters in the network???
        self.enc_final_pooling = nn.AdaptiveAvgPool2d(output_size=(1,1)) #x sofar [N x encoder_cout x 1 x 1]
        #make fc_mu and fc_logvar to make =========
        self.conv_mu = nn.Conv2d(encoder_cout, self.dim_latentvector,\
                                 kernel_size=(1, 1), stride=(1, 1), bias=True) #mu [N x dim_latentvector x 1 x 1]
        self.conv_logvar = nn.Conv2d(encoder_cout, self.dim_latentvector,\
                                     kernel_size=(1, 1), stride=(1, 1), bias=True) #logvar [N x dim_latentvector x 1 x 1]
        #make upchannel, the so that the decoder can take in z
        #self.dec_initial_upchannel = nn.Sequential(nn.ReLU(), nn.Conv2d(self.dim_latentvector, self.mod_decoder.list_upconvargs[0][0][0][0],\
                                                              #kernel_size=(1, 1), stride=(1, 1), bias=True)) #[N x 228 x 1 x 1]
        
    def encode(self, x):
        x = self.mod_encoder(x)  # [N x encoder_cout x H x W]
        x = self.enc_final_pooling(x) #[N x encoder_cout x 1 x 1] 
        mu = self.conv_mu(x) #[N x dim_latentvector x 1 x 1]
        logvar = self.conv_logvar(x) #[N x dim_latentvector x 1 x 1]
        return mu , logvar
        
    
    def reparameterize(self, mu, logvar):
        mu = mu[:,:,0,0] #[N x dim_latentvector]
        logvar = logvar[:,:,0,0] #[N x dim_latentvector]
        # ~ logvar = torch.ones_like(logvar)*np.log(1.0) #TODO: use fixed sigma???
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        z = z.unsqueeze(-1).unsqueeze(-1) #[N x dim_latentvector x 1 x 1]
        #z = self.dec_initial_upchannel(z) #[N x 228 x 1 x 1]
        return self.mod_decoder(z) #[N x 3 x 224 x 224]
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_reconst = self.decode(z)
        return x_reconst, z, mu, logvar
    
    def _infer_sizeof_encoderoutput(self):
        with torch.no_grad():
            x = torch.randn(1, self.input_channels, self.input_width, self.input_width)
            output_encoder = self.mod_encoder(x)
            return list(output_encoder.size())[1::]
            
    
    


