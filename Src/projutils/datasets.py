

import numpy as np
import matplotlib.pyplot as plt
import math
# import openslide
# import xml.etree.ElementTree as ET
import os
import sys
import time
import random
import csv
from skimage import data
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
import pickle
import openslide
list_pathstoadd = ["../../", "../../../PyDmed/"]
for path in list_pathstoadd:
    if(path not in sys.path):
        sys.path.append(path)
import projutils
import projutils.asap
import pydmed
from pydmed.utils.data import *
import pydmed.lightdl
from pydmed.lightdl import *
import pydmed.stat
from pydmed.stat import *


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


def visualize_incommingpatches(input_dataset, list_seenbatches,\
                               history_trainingloss, num_warmup_droppedpatches):
    #make np_toshow =======
    np_toshow = np.zeros((len(input_dataset.list_patients), len(list_seenbatches) ))
    for idx_batch in range(len(list_seenbatches)):
        list_idx_patientsofbatch = [input_dataset.list_patients.index(patient)\
                                     for patient in list_seenbatches[idx_batch]]
        np_toshow[list_idx_patientsofbatch, idx_batch] = 1.0
    #show the assignments =========
    figsize=(50,10)
    plt.figure(figsize=figsize)
    plt.imshow(np_toshow, cmap="gray", vmin=0.0, vmax=1.0)
    plt.axvline(x=num_warmup_droppedpatches, color="r", linewidth=2)
    plt.ylabel("index of patient", fontsize=18)
    plt.xlabel("index of mini-batch fed to the model", fontsize=18)
    plt.show()
    #show the history of training loss ===
    plt.figure(figsize=figsize)
    plt.plot(range(len(history_trainingloss)+num_warmup_droppedpatches),\
             num_warmup_droppedpatches*[history_trainingloss[0]]+history_trainingloss)
    plt.xlabel("iteration", fontsize=20)
    plt.ylabel("cross-entropy loss", fontsize=20)
    plt.show()
    #report the unseen patients =====
    print("{} patients are present in the datset"\
           .format(len(input_dataset.list_patients)))
    print("   {} patients were seen during training."\
           .format(
                np.sum(
                    (np.sum(np_toshow,1)>0.0)+0.0
                )
            ))



def get_rootpath_of_datasets():
    '''
    Returns the rootpath of datasets as saved in the file
    xxx.machineinfo.txt.
    '''
    path_repo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
    list_files = [u for u in os.listdir(path_repo)\
                  if(".machineinfo.txt" in u)]
    if(len(list_files) > 1):
        raise Exception("More than .machineinfo.txt files found:\n   {}".format(list_files)) 
    if(len(list_files) == 0):
        raise Exception("No .machineinfo.txt found in the root directory of the repo.")
    assert(len(list_files) == 1)
    fname_machineinfo = list_files[0]
    file_machineinfo = open(os.path.join(path_repo, fname_machineinfo), 'r')
    lines = file_machineinfo.readlines()
    dict_ds_to_rootpath = {}
    for line in lines:
         str_dsname, str_rootdir = line.split(",")
         if(str_rootdir[-1] == '\n'):
             str_rootdir = str_rootdir[0:-1] #to remove \n from the rootdir
         str_dsname = str_dsname.replace(" ", "")
         str_rootdir = str_rootdir.replace(" ", "")
         dict_ds_to_rootpath[str_dsname] = str_rootdir
    file_machineinfo.close()
    return dict_ds_to_rootpath
    

def otsu_get_foregroundmask(fname_wsi, scale_thumbnail):
    #settings =======
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


def get_warwickher2(idx_split, scale_foreground):
    rootdir_datasets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../Datasets/")
    rootpath_svsdataset = get_rootpath_of_datasets()["warwickher2"]
    #load the split
    fname_pkl = os.path.join(rootdir_datasets, "warwickher2/warwickher2.pkl")
    dict_pkl = pickle.load(open(fname_pkl, "rb"))
    ds_train = dict_pkl["split_{}".format(idx_split)]["ds_train"]
    ds_val = dict_pkl["split_{}".format(idx_split)]["ds_val"]
    ds_test = dict_pkl["split_{}".format(idx_split)]["ds_test"]


    #replace the rootdirs of records based on the local machininfo ====
    list_allpatients = ds_train.list_patients +\
                       ds_val.list_patients +\
                       ds_test.list_patients
    for patient in list_allpatients:
        for k in patient.dict_records.keys():
            curr_record = patient.dict_records[k]
            if(isinstance(curr_record, pydmed.utils.data.Record)):
                curr_record.rootdir = rootpath_svsdataset
    
    #presave paths to the foreground masks for the splits ====
    for patient in list_allpatients:
        try:
            fname_Her2 = os.path.join( 
                patient.dict_records["WSI_Her2"].rootdir,
                patient.dict_records["WSI_Her2"].relativedir
            )
            fname_xml = os.path.join( 
                patient.dict_records["XML_Her2"].rootdir,
                patient.dict_records["XML_Her2"].relativedir
            )
            np_foreground = projutils.asap.get_foreground_from_polyg(
                    fname_wsi = fname_Her2,\
                    fname_xml = fname_xml,\
                    scale = scale_foreground
                   )
            patient.dict_records["precomputed_polyongmask"] = np_foreground
        except Exception as e:
            print("failed.-------------")
            print(fname_Her2)
            print(fname_xml)
            print("------\n\n\n\n")
            print(str(e))
    #precompute foreground masks and polygonmasks ===========
    tstart_otsu = time.time()
    dict_patient_to_foreground = {}
    for idx_patient, patient in enumerate(list_allpatients):
        print(" computing foreground for patient {}".format(idx_patient))
        fname_wsi = os.path.join(patient.dict_records["WSI_Her2"].rootdir,\
                                 patient.dict_records["WSI_Her2"].relativedir)
        patient_foreground_mask =\
            otsu_get_foregroundmask(fname_wsi, scale_foreground)
        patient.dict_records["precomputed_otsu"] = patient_foreground_mask
        patient.dict_records["scale_thumbnail"] = scale_foreground
    tend_otsu = time.time()
    print("elapsed time = {}".format(tend_otsu - tstart_otsu))
    return ds_train, ds_val, ds_test
    
    
def get_privateher2(idx_split, scale_foreground):
    rootdir_datasets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../Datasets/")
    rootpath_svsdataset = get_rootpath_of_datasets()["privateher2"]
    #load the split
    fname_pkl = os.path.join(rootdir_datasets, "privateher2/privateher2.pkl")
    dict_pkl = pickle.load(open(fname_pkl, "rb"))
    ds_train = dict_pkl["split_{}".format(idx_split)]["ds_train"]
    ds_val = dict_pkl["split_{}".format(idx_split)]["ds_val"]
    ds_test = dict_pkl["split_{}".format(idx_split)]["ds_test"]


    #replace the rootdirs of records based on the local machininfo ====
    list_allpatients = ds_train.list_patients +\
                       ds_val.list_patients +\
                       ds_test.list_patients
    for patient in list_allpatients:
        for k in patient.dict_records.keys():
            curr_record = patient.dict_records[k]
            if(isinstance(curr_record, pydmed.utils.data.Record)):
                curr_record.rootdir = rootpath_svsdataset
    
    #presave paths to the foreground masks for the splits ====
    for patient in list_allpatients:
        try:
            fname_Her2 = os.path.join( 
                patient.dict_records["WSI_Her2"].rootdir,
                patient.dict_records["WSI_Her2"].relativedir
            )
            fname_xml = os.path.join( 
                patient.dict_records["XML_Her2"].rootdir,
                patient.dict_records["XML_Her2"].relativedir
            )
            np_foreground = projutils.asap.get_foreground_from_polyg(
                    fname_wsi = fname_Her2,\
                    fname_xml = fname_xml,\
                    scale = scale_foreground
                   )
            patient.dict_records["precomputed_polyongmask"] = np_foreground
        except:
            print("failed.-------------")
            print(fname_Her2)
            print(fname_xml)
            print("------\n\n\n\n")
    #precompute foreground masks and polygonmasks ===========
    tstart_otsu = time.time()
    dict_patient_to_foreground = {}
    for idx_patient, patient in enumerate(list_allpatients):
        print(" computing foreground for patient {}".format(idx_patient))
        fname_wsi = os.path.join(patient.dict_records["WSI_Her2"].rootdir,\
                                 patient.dict_records["WSI_Her2"].relativedir)
        patient_foreground_mask =\
            otsu_get_foregroundmask(fname_wsi, scale_foreground)
        patient.dict_records["precomputed_otsu"] = patient_foreground_mask
        patient.dict_records["scale_thumbnail"] = scale_foreground
    tend_otsu = time.time()
    print("elapsed time = {}".format(tend_otsu - tstart_otsu))
    return ds_train, ds_val, ds_test
