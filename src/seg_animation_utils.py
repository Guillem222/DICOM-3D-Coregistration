"""
File: seg_animation_utils.py
Author: Guillem
Date: April 27, 2024
Description: Medical Image Processing, Final Project, Objective 1, Sections C and D.
             Loads a segmentation image and CT image with PyDicom, then rearranges the image
             according to Slice location attribute, and finnaly it creates a rotating gif on
             the coronal-sagittal planes.
"""
import pydicom
import pydicom.datadict
import numpy as np
from matplotlib import pyplot as plt, animation
import os
import cv2
import matplotlib
import scipy




GLOBAL_CMAPS = {'copper': matplotlib.colormaps['copper'],
                'bone': matplotlib.colormaps['bone'],
                'tab10': matplotlib.colormaps['tab10'],
                'Set1': matplotlib.colormaps['Set1']}

def retrieve_data_from_folder_path(folder_path: str):
    """
    Given a folder ppath it retrieves all the imgs data and slices metadata required to build the 3d CT
    Returns: [(dcm px array 0,Slice Location 0), (dcm px array 1,Slice Location 1), ...], Slice thickness
                ([(np.ndarray,float),],float)
    """
    # List of tuples [(dcm px array 0,Slice Location 0), (dcm px array 1,Slice Location 1), ...]
    dcmArr_SliceLoc_list = [] 
    slice_thickness = 0.0

    # Iterate over all the files (dcm files) of the first study and first series

    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        if os.path.isfile(fpath) and fname[-3:]: # Check if its a DICOM file (just in case)
            dcm_file = pydicom.dcmread(fpath)
            slice_location = dcm_file.SliceLocation
            slice_thickness = dcm_file.SliceThickness
            dcmArr_SliceLoc_list.append((dcm_file.pixel_array,slice_location))

    return dcmArr_SliceLoc_list,slice_thickness



def retrieve_data_from_file_path(file_path: str):
    """
    Given a DICOM file path it retrieves its pixel information and its slice thickness.
    """
    dcm_file = pydicom.dcmread(file_path)
    slice_thickness = dcm_file.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].SliceThickness
    return dcm_file.pixel_array,slice_thickness


def extract_seg_slice_position(dcm_seg):
    """
    Given a DICOM file of a segmentation it retrieves, of each slice, the third coordinate of ImagePositionPatient and
    The reference number of the segmentation class the slice it belongs to.
    """
    slices_loc_seg = []
    for elem in dcm_seg.PerFrameFunctionalGroupsSequence:
            slices_loc_seg.append([ elem.PlanePositionSequence[0].ImagePositionPatient[2],
                                    elem.SegmentIdentificationSequence[0].ReferencedSegmentNumber])
            
    return slices_loc_seg


def arrange_segmentation(seg_array: np, seg_shape: tuple, ct_slice_loc: list, slices_loc_seg: list):
    """
    According to the SliceLocation of a CT image and the SliceLocation of a segmentation image,
    rearranges the slices of the segmentation image in order to match with the CT image.
    """

    seg_canvas = np.zeros(seg_shape)
    arr_ct_slice_location = np.array([elem[1] for elem in ct_slice_loc])

    arr_ct_slice_location = np.sort(arr_ct_slice_location)[::-1]

    for i in range(len(slices_loc_seg)):
        slice_loc, slice_class = slices_loc_seg[i]

        ct_loc = np.abs(arr_ct_slice_location - slice_loc).argmin()


        seg_canvas[ct_loc] = (seg_array[i] * slice_class)

    return seg_canvas


        

def rearrange_slices(dcmArr_SliceLoc_list: list):
    """
    Given a list [(dcm px array 0,Slice Location 0),...] it rearranges the slices of the
    list according to the slice location and returns a 3D reconstructed CT
    """

    # Sort the images according to the Slice Location
    dcmArr_SliceLoc_list = sorted(dcmArr_SliceLoc_list,key=lambda tp: tp[1],reverse=True)
    # Rearrange the images 
    slice_shape = dcmArr_SliceLoc_list[0][0].shape
    num_slices = len(dcmArr_SliceLoc_list)
    rearr_img = np.zeros((num_slices,slice_shape[0],slice_shape[1])) # shape -> (43,512,512)

    # Construct the 3D image
    for i,tp in enumerate(dcmArr_SliceLoc_list):
        rearr_img[i] = tp[0]

    return rearr_img


def mask_biggest_object(ct_img: np.ndarray):
    """
    We compute a mask of the patient's body over the axial plane.
    """
    # We filter the ct (only relevant pixels)
    filt_ct_img = ct_img.copy()
    filt_ct_img[filt_ct_img <= 0] = 0

    # We threshold the image though the axial planes
    axial_acum = np.mean(filt_ct_img,axis=0)

    _, binary_image = cv2.threshold(axial_acum, 180, 255,cv2.THRESH_BINARY)

    # We compute the largest contour (patients body mask)
    contours, _ = cv2.findContours(binary_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    mask = np.zeros_like(binary_image)
    cv2.drawContours(mask, [largest_contour], -1, 1, thickness=cv2.FILLED)

    return mask


def remove_artifact(ct_img: np.ndarray):
    """
    Remove "stretcher" artifact
    """

    

    body_mask = mask_biggest_object(ct_img)
    

    # The body mask is of 512x512, so we need to expanded in order to apply it to all the slices
    expanded_body_mask = np.expand_dims(body_mask,axis=0)

    # Apply the mask
    filt_ct_img = ct_img * expanded_body_mask

    return filt_ct_img


def colapse_segmentation_img(dcm__seg_img: np.ndarray, num_seg: int):
    """
    Combines all the classes/segmentations of a n separated segmentation image into a reduced colapsed version.
    """
    dcm_arr_shape = dcm__seg_img.shape
    # Prepare canvas to join all the segmentation classes
    seg_img = np.zeros((dcm_arr_shape[0] // 4, dcm_arr_shape[1], dcm_arr_shape[2]),dtype=np.int32)

    # Colapse all the segmentations into the canvas, asigning an incremental id for each class segmentation
    for i in range(num_seg):
        seg_img += dcm__seg_img[i * seg_img.shape[0]:(i + 1) * seg_img.shape[0],:,:] * (i + 1)

    return seg_img

def apply_windowing(img: np.ndarray,low: int,up: int, k: int = 255):
    """
    Apply windowing to given img
    HU scale
    Air: -1000 HU
    Water: 0 HU
    Bone: 50-1000 HU
    Metals: > 1000 HU
    """

    windowed_img = img.copy()
    windowed_img[windowed_img < low] = 0
    windowed_img[windowed_img > up] = k
    windowed_img = cv2.normalize(windowed_img, None, alpha=0, beta=k, norm_type=cv2.NORM_MINMAX)

    return windowed_img

def plot_axial_plane(img: np.ndarray, slice: int):
    """
    Plot an axial plane slice
    """
    plt.imshow(img[slice,:,:],cmap="gray")
    plt.title("Axial CT Slice")
    plt.show()


def MIP_sagittal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """
    Maximum Intensity Projection on the Sagittal Plane.
    """
    return np.max(img_dcm, axis=2)



def MIP_coronal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """
    Maximum Intensity Proejction on the Coronal Plane.
    """
    return np.max(img_dcm, axis=1)


def rotate_on_axial_plane(img_dcm: np.ndarray, angle_in_degrees: float) -> np.ndarray:
    """
    Rotate the image over the axial plane.
    """
    return scipy.ndimage.rotate(img_dcm, angle_in_degrees, axes=(1, 2), reshape=False)





def apply_cmap(img: np.ndarray, cmap_name: str = 'bone') -> np.ndarray:
    """ 
    Apply a colormap to a 2D image. 
    Extracted from: https://github.com/PBibiloni/11763/blob/activity02_solution/activity02.py
    """
    cmap_function = GLOBAL_CMAPS[cmap_name]
    return cmap_function(img)



def apply_alpha_fusion( dcm_img: np.ndarray,
                        seg_img: np.ndarray,
                        alpha: float,
                        dcm_cmap_name: str = 'bone',
                        seg_cmap_name: str = 'Set1' ):
    """
    Applies alpha fusion to two given images.
    """
    dcm_img_cmapped = apply_cmap(dcm_img, cmap_name = dcm_cmap_name)
    seg_img_cmapped = apply_cmap(seg_img, cmap_name = seg_cmap_name)
    seg_img_cmapped = seg_img_cmapped * seg_img[..., np.newaxis].astype('bool')


    # Apply alpha fusion
    return dcm_img_cmapped * (1-alpha) + seg_img_cmapped * alpha

def min_max_normalization(arr: np.array):
    """
    Applies min max normalization to a given array.
    """
    min_val = np.min(arr)
    max_val = np.max(arr)
    normalized_arr = (arr - min_val) / (max_val - min_val)
    return normalized_arr

def visualize_alpha_fusion(img: np.ndarray, mask: np.ndarray, alpha: float = 0.25):
    """
    Visualize both image and mask in the same plot.
    Debugging function.
    Extracted from: https://github.com/PBibiloni/11763/
    """
    
    img_sagittal_cmapped = apply_cmap(img, cmap_name='bone')    # Why 'bone'?
    mask_bone_cmapped = apply_cmap(mask, cmap_name='prism')     # Why 'prism'?
    mask_bone_cmapped = mask_bone_cmapped * mask[..., np.newaxis].astype('bool')


    fusioned_img = img_sagittal_cmapped * (1-alpha) + mask_bone_cmapped * alpha


    plt.imshow(fusioned_img, aspect=5)
    plt.title(f'Segmentation with alpha {alpha}')
    plt.show()




def create_axialRot_animation(  dcm_img: np.ndarray,
                                seg_img: np.ndarray,
                                save_path: str,
                                MIP_func,
                                n: int,
                                slice_thickness: float,
                                alpha_f_val: float =0.25,
                                dcm_cmap_name: str = 'bone',
                                seg_cmap_name: str = 'Set1',
                                margin_adjust: bool = True,
                                verbose: bool = True):
    
    """
    Creates an axial rotation animation GIF of a given DICOM image and a segmentation img.
    Extracted and modified from: https://github.com/PBibiloni/11763/blob/activity03_solution/activity03.py
    """
    
    if margin_adjust:
        # Match sizes (enlarge seg_img)
        margin = int((dcm_img.shape[0] - seg_img.shape[0]) / 2) # margin = 3
        dcm_img = dcm_img[margin:-margin]


    fig, ax = plt.subplots()

    #   Configure directory to save results
    os.makedirs(f'{save_path}MIP/', exist_ok=True)

    #   Create projections
    projections = []
    for idx, alpha in enumerate(np.linspace(0, 360*(n-1)/n, num=n)):
        rotated_img = rotate_on_axial_plane(dcm_img, alpha)
        rotated_seg = rotate_on_axial_plane(seg_img,alpha)

        projection_img = MIP_func(rotated_img) # Saggital or Coronal
        projection_seg = MIP_func(rotated_seg) # Saggital or Coronal

        fusioned_projection = apply_alpha_fusion(projection_img,projection_seg,alpha_f_val,dcm_cmap_name,seg_cmap_name)

        plt.imshow(fusioned_projection, aspect=slice_thickness)
        plt.savefig(f'{save_path}MIP/Projection_{idx}.png')      # Save animation
        plt.clf() # clean the buffer
        projections.append(fusioned_projection)  # Save for later animation
        if verbose == True:
            print(f"Frame: {idx}, stored in -> {save_path}MIP/Projection_{idx}.png")
    # Save and visualize animation
    animation_data = [
        [plt.imshow(img, vmin=0.0,vmax=1.0, animated=True, aspect=slice_thickness)]
        for img in projections
    ]

    
    anim = animation.ArtistAnimation(fig, animation_data,
                                interval=int((1/n) * 1000.0), blit=True) # n frames per second
    anim.save(f'{save_path}Animation.gif')  # Save animation
    plt.show()                              # Show animation








    