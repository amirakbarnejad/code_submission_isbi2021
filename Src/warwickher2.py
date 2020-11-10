
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from shapely.geometry import Polygon, Point
import sys
import os
import statistics
import psutil
import copy
import pickle
import re
from abc import ABC, abstractmethod
import math
import copy
import xml.etree.ElementTree as ET
import gc
from copy import deepcopy
from pathlib import Path
from skimage import data
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import skimage
import PIL
from skimage.filters import threshold_otsu
import torchvision.models as torchmodels
import openslide
import torch.utils.data
import torchvision
import torch.nn as nn
import pydmed
from pydmed.utils.data import *
import pydmed.lightdl
from pydmed.lightdl import *
import pydmed.stat
from pydmed.stat import *
import pydmed.extensions.dl
import projutils.datasets
import relatedwork
from relatedwork.utils.generativemodels import ResidualEncoder
import model
from model import *


def otsu_get_foregroundmask(fname_wsi, scale_thumbnail):
    #settings =======
#     scale_thumbnail =  0.01
    width_targetpatch = 5000 
    #extract the foreground =========================
    osimage = openslide.OpenSlide(fname_wsi)
    W, H = osimage.dimensions
    size_thumbnail = (int(scale_thumbnail*W), int(scale_thumbnail*H))
    pil_thumbnail = osimage.get_thumbnail(size_thumbnail)
    np_thumbnail = np.array(pil_thumbnail)
    np_thumbnail = np_thumbnail[:,:,0:3]
    np_thumbnail = rgb2gray(np_thumbnail)
    thresh = threshold_otsu(np_thumbnail)
    background = (np_thumbnail > thresh) + 0.0
    foreground = 1.0 - background
    w_padding_of_thumbnail = int(width_targetpatch * scale_thumbnail)
    foreground[0:w_padding_of_thumbnail, :] = 0
    foreground[-w_padding_of_thumbnail::, :] = 0
    foreground[: , 0:w_padding_of_thumbnail] = 0
    foreground[: , -w_padding_of_thumbnail::] = 0
    return foreground


def otsu_getpoint_from_foreground_in_polyg(fname_wsi, num_returned_points, patient):
    #settings =======
    scale_thumbnail =  patient.dict_records["scale_thumbnail"]
    np_inpoly = (patient.dict_records["precomputed_polyongmask"][:,:,0]>0.0)
    np_otsu = patient.dict_records["precomputed_otsu"]
    w = min(np_inpoly.shape[1], np_otsu.shape[1])
    h = min(np_inpoly.shape[0], np_otsu.shape[0])
    foreground = np_inpoly[0:h, 0:w] * np_otsu[0:h, 0:w]
    #select a random point =========================
    one_indices = np.where(foreground==1.0)
    i_oneindices, j_oneindices = one_indices[0].tolist(), one_indices[1].tolist()
    n = random.choices(range(len(i_oneindices)), k=num_returned_points)
    i_oneindices, j_oneindices = np.array(i_oneindices), np.array(j_oneindices)
    i_selected, j_selected = i_oneindices[n], j_oneindices[n]
    i_selected, j_selected = np.array(i_selected), np.array(j_selected)
    #     assert(foreground[i_selected, j_selected] == 1)
    i_selected_realscale, j_selected_realscale =\
        (i_selected/scale_thumbnail).astype(np.int), (j_selected/scale_thumbnail).astype(np.int)
    x, y = list(j_selected_realscale), list(i_selected_realscale)
    return x,y 


class WSIRandomBigchunkLoader(BigChunkLoader):
    @abstractmethod
    def extract_bigchunk(self, last_message_fromroot):
        '''
        Extract and return a bigchunk. 
        Please note that in this function you have access to
        self.patient and self.const_global_info.
        '''
        list_bigchunks = []
        num_bigpatches = 5
        
        #preselect `num_bigpatches` random points on foreground.
        wsi = self.patient.dict_records["WSI_Her2"]
        fname_wsi = os.path.join(wsi.rootdir, wsi.relativedir)
        all_randx, all_randy = \
            otsu_getpoint_from_foreground_in_polyg(fname_wsi, num_bigpatches, self.patient)
        
        for idx_bigpatch in range(num_bigpatches): #TODO:make tunable
            #settings ==== 
            flag_use_otsu = True
            #===
            wsi = self.patient.dict_records["WSI_Her2"]
            fname_wsi = os.path.join(wsi.rootdir, wsi.relativedir)
            osimage = openslide.OpenSlide(fname_wsi)
            w, h = self.const_global_info["width_bigchunk"],\
                   self.const_global_info["heigth_bigchunk"] 
            W, H = osimage.dimensions
            rand_x, rand_y = all_randx[idx_bigpatch],\
                             all_randy[idx_bigpatch]
            rand_x, rand_y = int(rand_x-(w*0.5)), int(rand_y-(h*0.5))
            
            pil_bigchunk = osimage.read_region([rand_x, rand_y], 1, [w,h])
            np_bigchunk = np.array(pil_bigchunk)[:,:,0:3]
            patient_without_foregroundmask = copy.deepcopy(self.patient)
            for k in patient_without_foregroundmask.dict_records.keys():
                if(k.startswith("precomputed")):
                    patient_without_foregroundmask.dict_records[k] = None
            bigchunk = BigChunk(data=np_bigchunk,\
                                 dict_info_of_bigchunk={"x":rand_x, "y":rand_y},\
                                 patient=patient_without_foregroundmask)
            #log to logfile
            # ~ self.log("new bigchunk with [left,top] = [{} , {}]\n".\
                     # ~ format(rand_x, rand_y))
            list_bigchunks.append(bigchunk)
        return list_bigchunks

class WSIRandomSmallchunkCollector(SmallChunkCollector):
    def __init__(self, *args, **kwargs):
        '''
        Inputs: 
            - mode_trainortest (in const_global_info): a strings in {"train" and "test"}.
                We need this mode because, e.g., colorjitter is different in training and testing phase.
        '''
        self.mode_trainortest = kwargs["const_global_info"]["mode_trainortest"]
        assert(self.mode_trainortest in ["train", "test"])
        #grab privates
        if(self.mode_trainortest == "train"):
            self.tfms_onsmallchunkcollection =\
                torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),\
                torchvision.transforms.ColorJitter(brightness=0,\
                                         contrast=0,\
                                         saturation=0.5,\
                                         hue=[-0.1, 0.1]),\
                torchvision.transforms.ToTensor(),\
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],\
                                                 std=[0.229, 0.224, 0.225])
            ])
        elif(self.mode_trainortest == "test"):
            self.tfms_onsmallchunkcollection =\
                torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),\
                torchvision.transforms.ToTensor(),\
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],\
                                                 std=[0.229, 0.224, 0.225])
            ])
        super(WSIRandomSmallchunkCollector, self).__init__(*args, **kwargs)
    
    @abstractmethod     
    def extract_smallchunk(self, call_count, list_bigchunks, last_message_fromroot):
        '''
        Extract and return a smallchunk. Please note that in this function you have access to 
        self.bigchunk, self.patient, self.const_global_info.
        Inputs:
            - list_bigchunks: the list of extracted bigchunks.
        '''"list_polygons"
        if(self.mode_trainortest == "test"):
            if(call_count > 100):
                return None
        bigchunk = random.choice(list_bigchunks)
        W, H = bigchunk.data.shape[1], bigchunk.data.shape[0]
        w, h = self.const_global_info["width_smallchunk"],\
               self.const_global_info["heigth_smallchunk"]
        rand_x, rand_y = np.random.randint(0, W-w), np.random.randint(0, H-h)
        np_smallchunk = bigchunk.data[rand_y:rand_y+h, rand_x:rand_x+w, :]
        #apply the transformation ===========
        if(self.tfms_onsmallchunkcollection != None):
            toret = self.tfms_onsmallchunkcollection(np_smallchunk)
            toret = toret.cpu().detach().numpy() #[3 x 224 x 224]
            toret = np.transpose(toret, [1,2,0]) #[224 x 224 x 3]
        else:
            toret = np_smallchunk
        #wrap in SmallChunk
        smallchunk = SmallChunk(data=np_smallchunk,\
                                dict_info_of_smallchunk={"x":rand_x, "y":rand_y},\
                                dict_info_of_bigchunk = bigchunk.dict_info_of_bigchunk,\
                                patient=bigchunk.patient)
        return smallchunk



class DLWithInTurnSched(LightDL):
    def __init__(self, *args, **kwargs):
        self.list_active_subclass = [] #push to left, pop from right
        self.list_waiting_subclass = []#push to right, pop from left
        self.sched_count_subclass = 0
        super(DLWithInTurnSched, self).__init__(*args, **kwargs)
    
    def schedule(self):
        '''
        This function is called when schedulling a new patient, i.e., loading a new BigChunk.
        This function has to return:
            - patient_toremove: the patient to remove, an instance of `utils.data.Patient`.
            - patient_toload: the patient to load, an instance of `utils.data.Patient`.
        In this function, you have access to the following fields:
            - self.dict_patient_to_schedcount: given a patient, returns the number of times the patients has been schedulled in dl, a dictionary.
            - self.list_loadedpatients:
            - self.list_waitingpatients:
            - TODO: add more fields here to provide more flexibility. For instance, total time that the patient have been loaded on DL.
        '''
        self.sched_count_subclass += 1
        #get initial fields ==============================
        list_loadedpatients = self.get_list_loadedpatients()
        list_waitingpatients = self.get_list_waitingpatients()
        waitingpatients_schedcount = [self.get_schedcount_of(patient)\
                                      for patient in list_waitingpatients]
        if(self.list_active_subclass == []):
            #the first call to `schedule function`
            self.list_active_subclass = list_loadedpatients
            self.list_waiting_subclass = list_waitingpatients
        #patient_toremove =======================
        patient_toremove = self.list_active_subclass[-1]
        #patient toadd ================
        patient_toload = self.list_waiting_subclass[0]
        #update the two in-turn lists
        self.list_active_subclass = [patient_toload] + self.list_active_subclass[0:-1]
        self.list_waiting_subclass = self.list_waiting_subclass[1::] + [patient_toremove]
        return patient_toremove, patient_toload
        
        
class WarwickHER2StatCollector(StatCollector):
    def __init__(self, module_pipeline, device, *args, **kwargs):
        #grab privates
        self.module_pipeline = module_pipeline
        self.device = device
        #make other initial operations
        self.module_pipeline.to(device)
        self.module_pipeline.eval()
        self.num_calls_to_getflagfinished = 0
        super(WarwickHER2StatCollector, self).__init__(*args, **kwargs)
        
    @abstractmethod
    def get_statistics(self, retval_collatefunc):
        x, list_patients, list_smallchunks = retval_collatefunc
        with torch.no_grad():
            netout = self.module_pipeline.testingtime_forward(x.to(self.device))#[32x100x7x7]
            netout = netout.cpu().numpy() #[32x100x7x7]
        list_statistics = []
        list_statistics += [Statistic(
                                stat=netout[n,:,:,:],\
                                source_smallchunk = list_smallchunks[n]
                              )
                             for n in range(netout.shape[0])]
        return list_statistics
    
    
    @abstractmethod
    def accum_statistics(self, prev_accum, new_stat, patient):
        if(prev_accum == None):
            #the first stat ====
            toret = {"count":1, "sum_encoded_descriptors": new_stat.stat}
        else:
            #not the first stat ====
            old_count, old_sum = prev_accum["count"], prev_accum["sum_encoded_descriptors"]
            MAXNUM_STATS = 500
            if(old_count < MAXNUM_STATS):
                toret = {"count": old_count+1,\
                         "sum_encoded_descriptors": old_sum + new_stat.stat}
            else:
                toret = {"count": old_count,\
                         "sum_encoded_descriptors": old_sum}
        return toret
                                 
    
    @abstractmethod
    def get_flag_finishcollecting(self):
        self.num_calls_to_getflagfinished += 1
        print("self.num_calls_to_getflagfinished = {}\n"\
                  .format(self.num_calls_to_getflagfinished))
        list_statcount = []
        for patient in self.dict_patient_to_accumstat.keys():
            if(self.dict_patient_to_accumstat[patient] != None):
                list_statcount.append(self.dict_patient_to_accumstat[patient]["count"])
            else:
                list_statcount.append(0)
        print(" numstats in [{} , {}],     num zeros = {}"
              .format(min(list_statcount) , max(list_statcount),
                      np.sum(np.array(list_statcount) == 0)) )
        #show bar of num-stats ====
        plt.figure()
        plt.bar(range(len(list_statcount)), list_statcount)
        plt.xticks(range(len(list_statcount)), rotation='vertical')
        plt.xlabel("index of patient")
        plt.ylabel("number of collected stats.")
        plt.show()
        MAXNUM_STATS = 500
        if((min(list_statcount)==MAXNUM_STATS) and (max(list_statcount)==MAXNUM_STATS)):
            if(True):#self.num_calls_to_getflagfinished > 100):
                return True
        else:
            return False
            
            
def confusion_matrix(pred_y, groundtruth_y):
    '''
    Computes confusion matrix.
    Inputs.
        - pred_y: a numpy array or list of containing the predicted labels.
        - groundtruth_y: a numpy array or list containing the groundtruth labels.
    '''
    #manage type variations ====
    if(isinstance(pred_y, list)):
        pass
    elif(isinstance(pred_y, np.ndarray)):
        pred_y = pred_y.flatten().tolist()
    if(isinstance(groundtruth_y, list)):
        pass
    elif(isinstance(groundtruth_y, np.ndarray)):
        groundtruth_y = groundtruth_y.flatten().tolist()
    #get possible labels ====
    list_possiblelabels = list(set(groundtruth_y + pred_y))
    conf_matrix = np.zeros((len(list_possiblelabels), len(list_possiblelabels)))
    assert(len(pred_y) == len(groundtruth_y))
    pred_y = [int(u) for u in pred_y]
    groundtruth_y = [int(u) for u in groundtruth_y]
    for idx_label in range(len(pred_y)):
        conf_matrix[groundtruth_y[idx_label], pred_y[idx_label]] =\
            conf_matrix[groundtruth_y[idx_label], pred_y[idx_label]]+1
    return conf_matrix

def warwickHer2_challengepoints(pred_y, groundtruth_y):
    '''
    The `points` used to evaluate teams in HER2 challenge.
    Inputs.
        - pred_y: a numpy array or list of containing the predicted labels.
        - groundtruth_y: a numpy array or list containing the groundtruth labels.
    '''
    if(isinstance(pred_y, np.ndarray)):
        pred_y = pred_y.flatten().tolist()
    conf_matrix = confusion_matrix(pred_y, groundtruth_y)
    np_scores = np.array([[15 , 15 , 10 , 0],\
                          [15  , 15  , 10 , 0],\
                          [2.5 , 2.5 , 15 , 5],\
                          [0 ,   0   , 10 , 15]])
    toret = np.sum(conf_matrix*np_scores)/len(pred_y)
    return toret
