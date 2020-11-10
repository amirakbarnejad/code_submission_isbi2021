

import numpy as np
import torch
import torch.nn as nn
import os
import sys
import abc
from abc import ABC, abstractmethod
import relatedwork
from relatedwork.utils.generativemodels import ResidualEncoder



def getassignment_img_to_wsi(x, list_patients, list_smallchunks):
    '''
    This function makes a batch usable for the module `akAvgPerWSI`.
    Inputs:
        - x: tensor of shape [N x 3 x H x W]
        - list_patients: list of `Patient`s.
        - list_smallchunks: list of `SmallChunk`s.
    Output:
        - tensor_list_assignmentindices: a tensor that assigns each sample
            in the batch to a group, as required by the module `akAvgPerWSI`.
    '''
    set_patients = set(list_patients)
    list_set_patients = list(set_patients)
    numgroups = len(list_set_patients)
    #make tensor_list_assignmentindices
    list_assignmentindices = [list_set_patients.index(patient)
                              for patient in list_patients]
    tensor_list_assignmentindices = torch.from_numpy(
                                    np.array(list_assignmentindices)
                                )
    list_different_groups = list_set_patients
    return tensor_list_assignmentindices, list_different_groups
    







class BoVWPipeline(nn.Module):
    def __init__(self, size_input,\
                 kwargs_stg1=None, kwargs_stg2=None, kwargs_stg3=None, kwargs_stg4=None):
        '''
        Inputs:
            - size_input: size of input to pipline, e.g., [32x3x224x224].
            - kwargs_stg1, kwargs_stg2, kwargs_stg3: three dictionaries, 
                    containing the kwargs to be passed to the stage modules. 
        '''
        super(BoVWPipeline, self).__init__()
        #grab privates ===
        self.kwargs_stg1 = kwargs_stg1
        self.kwargs_stg2 = kwargs_stg2
        self.kwargs_stg3 = kwargs_stg3
        self.kwargs_stg4 = kwargs_stg4
        #make the modules for different stages =======
        #make stage 1
        self.module_stg1 = self.get_stg1_descriptorgenerator(
                                    size_input = size_input,\
                                    kwargs_of_module = kwargs_stg1
                                )
        #make stage 2
        with torch.no_grad():
            x = torch.randn(*size_input)
            output_temp_stg1 = self.module_stg1(x)
            self.size_output_stg1 = list(output_temp_stg1.size())
        self.module_stg2 = self.get_stg2_theencoderdictionary(
                                    size_input = self.size_output_stg1,\
                                    kwargs_of_module = kwargs_stg2
                                )
        #make stage 3
        with torch.no_grad():
            _, output_temp_stg2 = self.module_stg2(output_temp_stg1)
            self.size_output_stg2 = list(output_temp_stg2.size())
        self.module_stg3 = self.get_stg3_descriptorpooling(
                                    size_input = self.size_output_stg2,\
                                    kwargs_of_module = kwargs_stg3
                                )
        #make stage 4
        with torch.no_grad():
            output_temp_stg3 = self.module_stg3(
                                      output_temp_stg2,\
                                      tensor_list_assignmentindices = torch.tensor([0])
                                    )
            self.size_output_stg3 = list(output_temp_stg3.size())
        self.module_stg4 = self.get_stg4_finalclassifier(
                                    size_input = self.size_output_stg3,\
                                    kwargs_of_module = kwargs_stg4
                                )
        
    @abstractmethod
    def get_stg1_descriptorgenerator(self, size_input, kwargs_of_module):
        '''
        This abstract method has to return a module corresponding to stage 1 of
        the pipeline. It can be, e.g., a fully convolutional network that produces
        [32x512x7x7] volumetric maps.
        Inputs:
            - size_input: a list like [32x3x224x224], the size of input to the module.
            - kwargs_of_module: the kwargs to be passed to the module, which 
                are passed when creating `BoVWPipeline`.
        '''
        pass
    
    @abstractmethod
    def get_stg2_theencoderdictionary(self, size_input, kwargs_of_module):
        '''
        This abstract method has to return a module corresponding to stage 2 of the pipeline.
        It can be, e.g., fisher vector encoder. The module produces, e.g., a [32x100x7x7]
        volumetric map.
        Inputs:
            - size_input: a list like [32x512x7x7], the size of input to the module.
            - kwargs_of_module: the kwargs to be passed to the module, which 
                are passed when creating `BoVWPipeline`.
        '''
        pass
    
    
    @abstractmethod
    def get_stg3_descriptorpooling(self, size_input, kwargs_of_module):
        '''
        This abstract method has to return a module corresponding to stage 3 of the pipeline.
        The module specifies how to pool the encoded descriptors to make a final vector as
        the encoded WSI. The module produces, e.g., a [32x 2000] volumetric map.
        - size_input: a list like [32x100x7x7], the size of input to the module.
        - kwargs_of_module: the kwargs to be passed to the module, which 
                    are passed when creating `BoVWPipeline`.
        '''
        pass
    
    @abstractmethod
    def get_stg4_finalclassifier(self, size_input, kwargs_of_module):
        '''
        This abstract method has to return a module corresponding to stage 4 of the pipeline,
        i.e., the final classifier.
        - size_input: a list like [32x100x1x1], the size of input to the module.
        - kwargs_of_module: the kwargs to be passed to the module, which 
                    are passed when creating `BoVWPipeline`.
        '''
        pass
    
    
    def forward(self, x, tensor_list_assignmentindices):
        '''
        Inputs:
            - x: batch of images, a tensor of shape, e.g., [10x3x224x224].
            - tensor_list_assignmentindices:
                    a list of indices that specifies to wchih WSI each image belongs. 
        '''
        output = self.module_stg1(x)
        _, output = self.module_stg2(output)
        output = self.module_stg3(output, tensor_list_assignmentindices)
        output = self.module_stg4(output)
        return output
        
    def testingtime_forward(self, x):
        '''
        This function returns the encoded descriptors of shape, .e.g, [32x100x7x7].
        The difference to the `forward` function is that this function does not apply
        descriptor pooling perWSI. Because in the test phase, pooling has to be done for "all" descriptors of a WSI.
        Inputs:
            - x: batch of images, a tensor of shape, e.g., [10x3x224x224].
            - tensor_list_assignmentindices:
                    a list of indices that specifies to wchih WSI each image belongs. 
        '''
        output = self.module_stg1(x)
        _, output = self.module_stg2(output)
        return output
        
    def getGradCAM(self, x, numclasses):
        '''
        Computes grad-cam for stage 1 outputs.
        Inputs.
            - x: the input batch, a tensor of shape, e.g., [32 x 3 x 224 x 224].
            - numclasses: an integer, the number of classes.
        Outputs.
            - toret_grads: a tensor containing the gradient of activations w.r.t. descriptors,
                           a tensor of shape, e.g., [numclasses x 32 x 1024 x 7 x 7].
            - toret_vals: a tensor containing the value of descriptors,
                           a tensor of shape, e.g., [32 x 1024 x 7 x 7].
        '''
        list_grads, list_values = [], []
        for idx_class in range(numclasses):
            self.zero_grad()
            #compute the required values ==========
            output_of_stg1 = self.module_stg1(x) #[32 x 1024 x 7 x 7]
            output_of_stg1.retain_grad()
            _, encoded_descriptors = self.module_stg2(output_of_stg1) #[32 x 1024*11*2 x 7 x 7], i.e., FV(x)
            output_activations = self.module_stg4(encoded_descriptors) #[32 x numclasses x 7 x 7]
            output_activations = torch.sum(
                                       torch.sum(output_activations, 3),
                                       2
                                    ) #[32 x numclasses]
            output_activations = torch.sum(output_activations[:, idx_class]) #scalar
            output_activations.backward()
            #compute Grad-CAM outputs ===========
            if(True):#with torch.no_grad():
                grad_descriptors_forclass = output_of_stg1.grad.cpu().numpy() + 0.0 #[32 x 1024 x 7 x 7]
                val_descriptors_forclass  = (output_of_stg1+0.0).detach().cpu().numpy() + 0.0
                list_grads.append(grad_descriptors_forclass)
                list_values.append(val_descriptors_forclass)
        toret_grads = np.stack(list_grads, 0)  #[numclasses x 32 x 1024 x 7 x 7]
        toret_vals  = np.stack(list_values, 0) #[numclasses x 32 x 1024 x 7 x 7] 
        return toret_grads, toret_vals[0,:,:,:,:]
        
class FishVectEncoder(nn.Module):
    def __init__(self, size_input, num_centers,\
                        pi_of_fishvect, sigma_of_fishvect):
        '''
        Inputs.
            - size_input: the size of the input to the module, e.g., [32x100x7x7].
            - num_centers: an integer, number of visual words.
            - pi_of_fishvect: float, the pi value of the FishVect, Eq.34 of the survey.
            - sigma_of_fishvect: float, the sigma value of the FishVect, Eq.34 of the survey.
        '''
        super(FishVectEncoder, self).__init__()
        #grab the privates =====
        self.num_centers = num_centers
        self.pi_of_fishvect = pi_of_fishvect
        self.sigma_of_fishvect = sigma_of_fishvect
        #make V, the visualwords =====
        NCHW = size_input
        dim_vectors = NCHW[1]
        self.V = torch.nn.Parameter(
                      torch.randn(self.num_centers, dim_vectors).\
                      unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
                    ) #[1 x 11 x 2048 x 1 x 1]
        
    
    def forward(self, x):
        '''
        Inputs.
            - x: volumetric tensor of shape [N x 2048 x 94 x 94]
        Outputs.
            - soft_assignment: [N x 11 x 94 x 94]
            - FV_mu_part: [N x 11*2048  x 94 x 94]
            - FV_sigma_part: #[N x 11*2048  x 94 x 94]
        '''
        #make tao (Eq.35 of the survey) ===================
        x_resized = x.unsqueeze(1) #[N x 1 x 2048 x 94 x 94]
        x_min_V = x_resized - self.V.detach() #[N x 11 x 2048 x 94 x 94]
        tao_exponent = torch.sum(
                        -(x_min_V*x_min_V)/(2.0*self.sigma_of_fishvect*\
                                                self.sigma_of_fishvect),\
                        2
                    ) #[N x 11 x 94 x 94]
        soft_assignment = torch.nn.functional.softmax(
                                    tao_exponent,\
                                    1
                                  ) #[N x 11 x 94 x 94]
        #make FV_mu_part =======
        FV_mu_part = (x_min_V/self.sigma_of_fishvect) *\
                      soft_assignment.unsqueeze(2)*\
                      (1.0/np.sqrt(self.pi_of_fishvect))#[N x 11 x 2048 x 94 x 94]
        list_dim = list(FV_mu_part.size())
        new_list_dim = [list_dim[0]] + [-1] + list_dim[3:5]
        FV_mu_part = FV_mu_part.view(*new_list_dim) #[N x 11*2048  x 94 x 94]
        #make FV_sigma_part ======
        FV_sigma_part =\
           ((x_min_V/self.sigma_of_fishvect)*\
            (x_min_V/self.sigma_of_fishvect) - 1) *\
             soft_assignment.unsqueeze(2)*\
             (1.0/np.sqrt(2.0*self.pi_of_fishvect))#[N x 11 x 2048 x 94 x 94]
        FV_sigma_part = FV_sigma_part.view(*new_list_dim) #[N x 11*2048 x 94 x 94]
        encoded_descriptors = torch.cat([FV_mu_part, FV_sigma_part],\
                                        1) #[N  x  2*11*2048  x  94  x  94]
#         FV_sigma_part = FV_sigma_part.view(*new_list_dim) #[N x 11*2048 x 94 x 94]
        #make encoded_descriptors ======
#         encoded_descriptors = torch.cat([FV_mu_part, FV_sigma_part],\
#                                         1) #[N  x  2*11*2048  x  94  x  94]
#         encoded_descriptors = torch.sum(encoded_descriptors, 3) #[N  x  2*11*2048  x  94]
#         encoded_descriptors = torch.sum(encoded_descriptors, 2) #[N  x  2*11*2048]
#         encoded_descriptors = encoded_descriptors/(new_list_dim[-1]*new_list_dim[-2])
        return soft_assignment, encoded_descriptors
                              
        
#make AvgPerWSI layer ================
class AvgPoolVectorsPerWSI(nn.Module):
    def __init__(self, size_input, device):
        '''
        Inputs:
            - size_input: size of the input, e.g., [32 x 2000 x 7 x 7].
        '''
        super(AvgPoolVectorsPerWSI, self).__init__()
        #grab privates
        self.size_input = size_input
        self.device = device
        
    def forward(self, x, tensor_list_assignmentindices):
        '''
        Inputs:
            - x: tensor of shape [NxMxHxW], the M-dimensional encoder descriptors.
            - tensor_list_assignmentindices: 
                 A list containing the assignment of each 
                        instnace in the batch to a group.
                 The length of this list must equal to batch_size.
                 The minimum value of this list has to be 0. 
                 Indeed, max(list_assignmentindices) is equal to the number of different groups
                     to which the batch's instances are assigned.
        Output:
            - a tensor of shape [numgroups x M].
             Each row is the average `encoded(descriptor)`s of a specific group.
             Each group can correspond to, e.g., a specific WSI.
        '''
        #make constants
        N, M, H, W = self.size_input
        #handle the case where all patches are from a single WSI
        numgroups = int(torch.max(tensor_list_assignmentindices))+1
        if(numgroups == 1):
            toret = torch.sum(x, dim=3) #[NxMxH]
            toret = torch.sum(toret, dim=2) #[NxM]
            toret = torch.sum(toret, dim=0) #[M] 
            toret = toret/(N*H*W) #[M]
            toret = toret.unsqueeze(-1).unsqueeze(-1).unsqueeze(0) #[1xMx1x1]
            return toret
        #handle the case where patches are from different WSIs.
        with torch.no_grad():
            numgroups = int(torch.max(tensor_list_assignmentindices))+1
            freqof_groups = [tensor_list_assignmentindices.cpu().numpy().tolist().count(n)
                             for n in range(numgroups)]
            freqof_groups = torch.from_numpy(np.array(freqof_groups)) #[numgroups]
            freqof_groups = freqof_groups.unsqueeze(1).to(self.device)
                             #[numgroups x 1]
        toret = torch.zeros((numgroups, M, H, W)).to(self.device)
        toret.index_add_(0,
                    tensor_list_assignmentindices,
                    x) #[numgroups x M x H x W]
        toret = torch.sum(toret, dim=3) #[numgroups x M x H]
        toret = torch.sum(toret, dim=2) #[numgroups x M]
        toret = toret/(freqof_groups*H*W) #[numgroups x M]
        toret = toret.unsqueeze(-1).unsqueeze(-1) #[numgroups x M x 1 x 1]
        return toret

