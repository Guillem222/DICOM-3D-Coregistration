"""
File: main_rigid_coregistration.py
Author: Guillem
Date: May 12, 2024
Description: Medical Image Processing, Final Project, Objective 2
             Image Coregistration of two brains. Involves loading and rearranging both a reference
             brain and a patient's input brain from DICOM files into 3D images (ndarrays). Through an optimization
             optimization function, the input brain is transformed iteratively to achieve an optimal alignment with
             the reference brain. Ultimately, the thalamus region is located within the patient's brain.
"""

import pydicom
from rigid_coregistration_utils import *

DATA_COREG_PATH = "../data_coregistration/"
RM_BRAIN_3D_SPGR_PATH = "RM_Brain_3D-SPGR/"
PHANTOM_FNAME = "icbm_avg_152_t1_tal_nlin_symmetric_VI.dcm"
ATLAS_FNAME = "AAL3_1mm.dcm"
PATIENT_0_DCM_FILE = '000050.dcm'
ATLAS_INFO_PATH = '../data_coregistration/AAL3_1mm.txt'


if __name__ == "__main__":

    
    SEARCH_OPTIMIZER = False
    COREGISTER = False
    VISUAL_CHECK_COREG = False
    CREATE_AXIAL_ROT_GIF = False

    
    # LOAD PATIENT BRAIN (DCM FILE)


    folder_path = DATA_COREG_PATH + RM_BRAIN_3D_SPGR_PATH
    

    dcm_0_input_patient = pydicom.dcmread(folder_path + "/" + PATIENT_0_DCM_FILE)
    # Pixel Spacing. (0028,0030) 1C. Physical distance in the Patient between the center of each pixel,
    # specified by a numeric pair - adjacent row spacing (delimiter) adjacent column spacing in mm.
    print("Pixel Spacing Input Patient: ",dcm_0_input_patient.PixelSpacing)
    pixel_spacing = dcm_0_input_patient.PixelSpacing

    dcmArr_SliceLoc_list,slice_thickness = retrieve_data_from_folder_path(folder_path)

    ct_img = rearrange_slices(dcmArr_SliceLoc_list)



    print("CT original SliceThickness: ",slice_thickness)
    print("CT Original Size: ", ct_img.shape)


    # LOAD PHANTOM
    file_path_phtantom = DATA_COREG_PATH + PHANTOM_FNAME
    dcm_panthom = pydicom.dcmread(file_path_phtantom)
    phantom_img = dcm_panthom.pixel_array
    print("Original Phantom Shape: ",phantom_img.shape)
    
    # LOAD ATLAS
    file_path_atlas = DATA_COREG_PATH + ATLAS_FNAME
    dcm_atlas = pydicom.dcmread(file_path_atlas)
    atlas_img = dcm_atlas.pixel_array
    print("Atlas Shape: ",atlas_img.shape)


    # MATCH ATLAS AND PHANTOM SHAPES
    phantom_img = crop_img(phantom_img,atlas_img.shape)
    

    print("Pixel Spacing Phantom: ",dcm_panthom.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing)
    print("New phantom Size: ",phantom_img.shape)

   

    # MATCH CT_IMG WITH PHANTOM IMG

    
    ct_img = resize_3d_image(ct_img,(int(ct_img.shape[1] * pixel_spacing[0]), int(ct_img.shape[2] * pixel_spacing[1])))
    print("Resized ct image shape:", ct_img.shape)
    ct_img = crop_img(ct_img,phantom_img.shape)
    print("New ct image shape: ",ct_img.shape)
    


    # NORMALIZE IMGS

    ct_img = min_max_normalization(ct_img) # interval [0,1]
    phantom_img = min_max_normalization(phantom_img) # interval [0,1]


    # COREGISTER PHANTOM AND INPUT 

    # Search best optimization method

    

    if SEARCH_OPTIMIZER:
        opt_methods = ['L-BFGS-B','BFGS','Nelder-Mead','Powell','CG','SLSQP','COBYLA','TNC']
        search_best_optimizer(opt_methods, ct_img, phantom_img)

    

    optimal_parameters = []
    if COREGISTER:
        result = coregister_imgs(phantom_img,ct_img,opt_method='Nelder-Mead')
        print(result.x)
        optimal_parameters = result.x
        new_img = apply_transformation(ct_img,phantom_img,optimal_parameters)
        
        visualize_mid_slices(new_img,phantom_img)

    else:
        optimal_parameters = [-1.81224448e+02, -2.03386103e+02,  6.95491340e-05, # Translation vector
                              2.97892138e+00, # Rotation Angle in rads
                              -3.21520452e-06, -3.67973836e-05, -1.05940707e+00] # Axis of rotation
        
    

    # CHECK VISUALLY CORRECTNESS OF COREGISTRATION

    
    new_img = apply_transformation(ct_img,phantom_img,optimal_parameters)
    if VISUAL_CHECK_COREG:
        visualize_mid_slices(new_img,phantom_img)

    # Create gif
    #create_axialRot_animation_v3(new_img,phantom_img,'coregistration_results/',MIP_sagittal_plane,30,1,0.3,True)
    

    # FIND THALAMUS REGION
    
    # Obtain thalamus indices
    
    thalamus_input_space = thalamus_2_input_space(ATLAS_INFO_PATH,atlas_img,optimal_parameters)



    visualize_mid_slices(thalamus_input_space,ct_img,mid_coronal=125,mid_axial=90,mid_sagittal=78,alpha=0.7,inp_cmap='tab10',ph_cmap='bone')
    
    if CREATE_AXIAL_ROT_GIF:
        create_axialRot_animation(thalamus_input_space,ct_img,'thalamus_ct_img_t/',MIP_sagittal_plane,30,1,0.7,'tab10','bone',False)