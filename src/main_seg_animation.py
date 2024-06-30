"""
File: seg_animation_utils.py
Author: Guillem
Date: May 12, 2024
Description: Medical Image Processing, Final Project, Objective 1, Sections C and D.
             Loads a segmentation image and CT image with PyDicom, then rearranges the image
             according to Slice location attribute, and finnaly it creates a rotating gif on
             the coronal-sagittal planes.
"""

# Constant variables for the dicom files location
PATIENT_PATH = "../dicom_liver_files/"
FIRST_STUDY_PATH = "2.000000-PRE LIVER-76970/"
SEG_FOLDER = "300.000000-Segmentation-99942/"
SEG_NAME = "1-1.dcm"


import pydicom
from seg_animation_utils import *


if __name__ == "__main__":

    
    # LOAD SEGMENTATION
    NUM_SEG = 4

    file_path = PATIENT_PATH + SEG_FOLDER + SEG_NAME
    dcm_file = pydicom.dcmread(file_path)

    dcm_seg_img = dcm_file.pixel_array

    seg_img =colapse_segmentation_img(dcm_seg_img,4)

    
    # LOAD CT IMAGE
    folder_path = PATIENT_PATH + FIRST_STUDY_PATH

    dcmArr_SliceLoc_list,slice_thickness = retrieve_data_from_folder_path(folder_path)

    

    ct_img = rearrange_slices(dcmArr_SliceLoc_list)


    ct_img = remove_artifact(ct_img)

    ct_img = min_max_normalization(ct_img)
    ct_img[ct_img < 0.3] = 0 # We remove non relevant parts

    create_axialRot_animation(ct_img,seg_img,"alpha_fusion_results_v4/",MIP_sagittal_plane,30,5,0.4)

    
