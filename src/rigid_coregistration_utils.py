"""
File: rigid_coregistration_utils.py
Author: Guillem
Date: April 30, 2024
Description: Medical Image Processing, Final Project, Objective 2
             Image Coregistration of two brains. Involves loading and rearranging both a reference
             brain and a patient's input brain from DICOM files into 3D images (ndarrays). Through an optimization
             optimization function, the input brain is transformed iteratively to achieve an optimal alignment with
             the reference brain. Ultimately, the thalamus region is located within the patient's brain.
"""
import pydicom
import pydicom.datadict
import numpy as np
from matplotlib import pyplot as plt
import os
import cv2
import matplotlib
import cv2
import time
import math
import pandas as pd
import quaternion
from scipy.optimize import minimize
from seg_animation_utils import *







def crop_img(img: np.ndarray, resemble_shape: tuple):
    """
    Crops a given image using a desired shape.
    """
    crop_axis = np.zeros(len(resemble_shape), dtype=np.int32)
    decr_dim = np.zeros(len(resemble_shape), dtype=np.int32) # to adjust non divisible by 2 dimensions

    for i in range(len(crop_axis)):
        crop_axis[i] = int((img.shape[i] - resemble_shape[i]) / 2)
        if ((img.shape[i] - resemble_shape[i]) % 2) != 0:
            decr_dim[i] = 1

    img = img[  crop_axis[0]:(-crop_axis[0] - decr_dim[0]),
                crop_axis[1]:(-crop_axis[1] - decr_dim[1]),
                crop_axis[2]:(-crop_axis[2] - decr_dim[2])]
    
    return img

def resize_3d_image(image_stack: np.ndarray, new_size: tuple):
    """
    It resizes all the slices of a 3D image
    """
    resized_stack = []
    for img in image_stack:
        resized_img = cv2.resize(img, new_size)
        resized_stack.append(resized_img)
    return np.array(resized_stack)


def plot_axial_plane(img: np.ndarray, slice: int, num_fig: int):
    """
    Plots an axial slice of a given image
    """
    plt.figure(num_fig)
    plt.imshow(img[slice,:,:],cmap="gray")
    plt.title("Axial CT Slice")


def translation(img_indices: np.ndarray, translation_vector: tuple[float,float,float]):
    """
    Applies a 3D translation to the (x,y,z) coordinates (indices) of a 3D image. 
    """
    # Ensure type
    if isinstance(translation_vector, list) or isinstance(translation_vector,tuple):
        translation_vector = np.array(translation_vector)

    # Apply translation
    img_indices += translation_vector.astype(np.int32)
    return img_indices



def axial_rotation(img_indices: np.ndarray, angle_in_rads: float, axis_of_rotation: tuple[float,float,float]):
    """
    Applies a axial rotation, defined by the given parameters, to the (x,y,z) coordinates (indices) of a 3D image.
    It uses a quaternions approach.
    """
    v1, v2, v3 = axis_of_rotation
    # Normalize axis of rotation to avoid restrictions on optimizer
    v_norm = math.sqrt(sum([coord ** 2 for coord in [v1, v2, v3]]))
    v1, v2, v3 = v1 / v_norm, v2 / v_norm, v3 / v_norm


    # Quaternion associated to axial rotation.
    cos, sin = math.cos(angle_in_rads / 2), math.sin(angle_in_rads / 2)
    q_ax_rot = np.quaternion(cos,sin * v1, sin * v2, sin * v3)



    # Convert those points to quaterions q(0,x,y,x)
    q_indices = np.hstack((np.zeros((img_indices.shape[0], 1)), img_indices))
    q_indices = quaternion.as_quat_array(q_indices.astype(float))

    # Conjugate quaternion associated to axial rotation
    
    q_star = np.quaternion.conjugate(q_ax_rot)

    # Apply the transformation (axial rotation)



    
    q_tmp = q_indices * q_star
    q_prime = q_ax_rot * q_tmp

    

    # Convert to 3d point

    p_transf = quaternion.as_float_array(q_prime)[:,1:].astype(np.int32)


    return p_transf



def translation_then_axialrotation(img_indices: np.ndarray, parameters: tuple[float,...]):
    """
    Applies a translation and then an axial rotation to the (x,y,z) coordinates (indices) of a 3D image.
    """
    t1, t2, t3, angle_in_rads, v1, v2, v3 = parameters


    p_transf_indices = translation(img_indices,[t1,t2,t3])
    p_transf_indices = axial_rotation(p_transf_indices,angle_in_rads,[v1,v2,v3])
    
    return p_transf_indices


def axial_rotation_then_translation(img_indices: np.ndarray, parameters: tuple[float,...]):
    """
    Applies an axial rotation and then a translation to the (x,y,z) coordinates (indices) of a 3D image.
    """
    t1, t2, t3, angle_in_rads, v1, v2, v3 = parameters

    p_transf_indices = axial_rotation(img_indices,angle_in_rads,[v1,v2,v3])
    p_transf_indices = translation(p_transf_indices,[t1,t2,t3])
    
    
    return p_transf_indices




def squared_errors(ref_img: np.ndarray, inp_img: np.ndarray) -> np.ndarray:
    """
    Computes the Squared Errors between two images.
    """
    return ((ref_img-inp_img)**2).ravel()




def coregister_imgs(ref_img: np.ndarray, inp_img: np.ndarray, opt_method: str):
    """
    Coregister two images using an optimization algorithm of the scipy minimize method.
    """
    

    initial_parameters = compute_initial_parameters(ct_shape=inp_img.shape)

    print("Initial Parameters: ",initial_parameters)

    
    # Create a canvas 
    trans_canvas = np.zeros_like(inp_img)

    # Get all the indices from canvas
    canvas_indices = np.argwhere(trans_canvas >= 0)


    def function_to_minimize(parameters: tuple[float,...]):
        """
        Function that uses the inverse transform approach (from canvas to input),
        to transform the input image with a translation and then axial rotation with
        the given parameters. Then, creates the transformated image and returns the
        MSE with the reference image for the optimizier to handle it.
        """
        # Transform the indices (x,y,z)
        inv_params = inverse_parameters(parameters)
        p_transf_indices = axial_rotation_then_translation(canvas_indices,inv_params)

        # Create the transformed CT image with the transformed indices and the original canvas indices
        trans_inp_img = canvas_indices_2_trans_img(inp_img,trans_canvas,p_transf_indices,canvas_indices)
        
        # Compute SE
        vec_res = squared_errors(ref_img,trans_inp_img)

        # Compute MSE
        return np.mean(vec_res)

    
    result = minimize(fun=function_to_minimize,
                      x0=initial_parameters,
                      method=opt_method,
                      tol=1e-6)
    
   

    return result



def filter_indices(indices: np.ndarray, shape: tuple):
    """
    Filters the indices in order they to do not fall outside the given shape.
    """
    axis0_range = [0, shape[0]]
    axis1_range = [0, shape[1]]
    axis2_range = [0, shape[2]]

    # Create boolean masks for each axis based on the ranges
    mask_axis0 = (indices[:, 0] >= axis0_range[0]) & (indices[:, 0] < axis0_range[1])
    mask_axis1 = (indices[:, 1] >= axis1_range[0]) & (indices[:, 1] < axis1_range[1])
    mask_axis2 = (indices[:, 2] >= axis2_range[0]) & (indices[:, 2] < axis2_range[1])

    # Combine the masks to get the final mask for filtering
    final_mask = mask_axis0 & mask_axis1 & mask_axis2

    # Filter the indices based on the final mask
    filtered_indices = indices[final_mask]

    return filtered_indices,final_mask



def canvas_indices_2_trans_img(ct_img: np.ndarray, canvas: np.ndarray, p_transf_indices: np.ndarray, canvas_indices: np.ndarray):

    """
    Assigns the values of the transformed indices from the input image to the
    transformed canvas.
    """
    # FILTER THE INDICES AND REMOVE THE ELIMINATED POSITIONS IN img variable
    # Indices will remain from size (n,3) and filt values from size (n)
    
    p_transf_filt, mask = filter_indices(p_transf_indices,ct_img.shape)
    
    # Filter the original canvas indices (in order to not try to fill indices that are outside the image)

    filt_canvas_indices = canvas_indices[mask]

    # CANVAS[Original Filtered Indices] = CT_IMG[Trasformed Filtered Images]
    canvas[filt_canvas_indices[:,0],filt_canvas_indices[:,1],filt_canvas_indices[:,2]] = ct_img[p_transf_filt[:,0],p_transf_filt[:,1],p_transf_filt[:,2]]

    return canvas



def compute_initial_parameters(ct_shape: tuple):
    """
    Creates the initial parameters in the form of:
    [t1,t2,t3,angle,v1,v2,v3] Translation vector (t1,t2,t3), rotation angle (angle) and
    Axis of rotation (v1,v2,v3).
    The initial rotations and translations are hardcoded. With visual inspections I saw that
    the input brain needed to be rotated 180º in two axis. Therefore, those would be our
    initial parameters in order not to fall into a local minima.
    """

    # Create translation and rotation matrices
    T1 = np.array([ [1, 0, 0, -ct_shape[0]],
                    [0, 1, 0, 0],
                    [0, 0, 1, -ct_shape[2]],
                    [0, 0, 0, 1]])

    T2 = np.array([ [1, 0, 0, 0],
                    [0, 1, 0, -ct_shape[1]],
                    [0, 0, 1, -ct_shape[2]],
                    [0, 0, 0, 1]])
    
    R1 = np.array([ [math.cos(math.pi), 0, math.sin(math.pi), 0],
                    [0, 1, 0, 0],
                    [-math.sin(math.pi), 0, math.cos(math.pi), 0],
                    [0, 0, 0, 1]])
    
    R2 = np.array([ [1, 0, 0, 0],
                    [0, math.cos(math.pi), -math.sin(math.pi), 0],
                    [0, math.sin(math.pi), math.cos(math.pi), 0],
                    [0, 0, 0, 1]])

    # Combine transformations (T1 * R1 * T2 * R2)
    combined_matrix = np.dot(np.dot(np.dot(T1, R1),T2),R2)

    # Extract translation vector and rotation submatrix
    translation = combined_matrix[:3, 3]
    rotation_matrix = combined_matrix[:3, :3]

    # Convert rotation matrix to rotation axis plus rotation angle
    axis_initial_parameters = rotation_mat_2_rot_axis(rotation_matrix)

    # Combine translation vector and axis rotation parameters
    initial_parameters = list(translation) + axis_initial_parameters

    return initial_parameters

def rotation_mat_2_rot_axis(rotation_matrix: np.ndarray):
    """
    Converts a rotation matrix to the form [Angle of rotation, axis of rotation]
    """
    # Calculate rotation angle and axis
    angle = math.acos((np.trace(rotation_matrix) - 1) / 2)
    axis = [
        rotation_matrix[2, 1] - rotation_matrix[1, 2],
        rotation_matrix[0, 2] - rotation_matrix[2, 0],
        rotation_matrix[1, 0] - rotation_matrix[0, 1]
    ]
    axis_norm = np.linalg.norm(axis)
    if axis_norm != 0:
        axis = [x / axis_norm for x in axis]

    # Convert axis to initial_parameters format
    return [angle, axis[0], axis[1], axis[2]]

    


def apply_transformation(   ct_img: np.ndarray,
                            phantom_img: np.ndarray,
                            parameters: tuple[float,...],
                            plot: bool = False,
                            slice_: int =100):
    """
    Applies a translation and then axial rotation (using the inverse apporach).
    """
    trans_canvas = np.zeros_like(ct_img)

    indices = np.argwhere(trans_canvas >= 0) # all the indices

    
    inv_params = inverse_parameters(parameters)
    p_transf_indices = axial_rotation_then_translation(indices,inv_params)

    new_img = canvas_indices_2_trans_img(ct_img,trans_canvas,p_transf_indices,indices)

    if plot:
        plot_axial_plane(new_img,slice_,1)
        plot_axial_plane(phantom_img,slice_,2)
        plt.show()

    return new_img


def visualize_mid_slices( ct_img: np.ndarray,
                                    phantom_img: np.ndarray,
                                    alpha: int = 0.3,
                                    mid_axial: int = None,
                                    mid_coronal: int = None,
                                    mid_sagittal: int = None,
                                    inp_cmap: str = 'copper',
                                    ph_cmap: str = 'bone'):
    """
    Visualizes three slices (of axial, coronal and sagittal planes) from the alpha fusion
    result of the reference phantom image and the transformed input patient's brain.
    """

    if mid_axial is None:
        mid_axial = ct_img.shape[0] // 2
    
    if mid_coronal is None:
        mid_coronal = ct_img.shape[1] // 2
    
    if mid_sagittal is None:
        mid_sagittal = ct_img.shape[2] // 2

    # Plot middle axial plane
    ct_axial_slice = ct_img[mid_axial,:,:]
    ph_axial_slice = phantom_img[mid_axial,:,:]
    fusion_axial_slice = apply_alpha_fusion(ct_axial_slice,ph_axial_slice,alpha,inp_cmap,ph_cmap)

    # Plot middle coronal plane
    ct_coronal_slice = ct_img[:,mid_coronal,:]
    ph_coronal_slice = phantom_img[:,mid_coronal,:]
    fusion_coronal_slice = apply_alpha_fusion(ct_coronal_slice,ph_coronal_slice,alpha,inp_cmap,ph_cmap)

    # Plot middle sagittal plane
    ct_sagittal_slice = ct_img[:,:,mid_sagittal]
    ph_sagittal_slice = phantom_img[:,:,mid_sagittal]
    fusion_sagittal_slice = apply_alpha_fusion(ct_sagittal_slice,ph_sagittal_slice,alpha,inp_cmap,ph_cmap)


    plt.figure(1)
    plt.imshow(fusion_axial_slice)
    plt.title("Axial Coregistration Slice")

    plt.figure(2)
    plt.imshow(fusion_coronal_slice)
    plt.title("Coronal Coregistration Slice")

    plt.figure(3)
    plt.imshow(fusion_sagittal_slice)
    plt.title("Sagittal Coregistration Slice")

    plt.show()



def inverse_parameters(parameters: tuple[float,...]):
    """
    Inverse the parameters of a "translation and then axial rotation" parameters.
    """
    t1, t2, t3, angle_in_rads, v1, v2, v3 = parameters
    return (-t1,-t2,-t3,angle_in_rads,-v1,-v2,-v3)


def min_max_normalization(arr: np.ndarray):
    """
    Applies a min max normalization of a vector
    """
    min_val = np.min(arr)
    max_val = np.max(arr)
    normalized_arr = (arr - min_val) / (max_val - min_val)
    return normalized_arr



def thalamus_ids(atlas_info_path: str):
    """
    Extract the IDs corresponding to the thalamus in the Atlas img.
    """
    df = pd.read_csv(atlas_info_path, sep=' ')
    filt_df = df[df.iloc[:, 1].str.contains('Thal')]
    return np.array(filt_df.iloc[:,0])


def thalamus_2_input_space(atlas_info_path: str, atlas_img: np.ndarray, parameters: tuple[float,...]):
    """
    Transform the thalamus to the patient's input brain space.
    """
    # Get thalamus ids from atlas information file
    thal_ids = thalamus_ids(atlas_info_path)

    # Mask of all the thalamus region
    mask = np.isin(atlas_img,thal_ids)

    # Transform mask to patient's brain space

    trans_canvas = np.zeros_like(atlas_img)

    indices = np.argwhere(trans_canvas >= 0) # all the indices

    # Inverse twice
    #inv_params = inverse_parameters(optimal_parameters) -> First inverse because we need the inverse operation
    #inv_params = inverse_parameters(inv_params) -> Second Inverse because we use the reversed search apporach 
    # (Ask each pixel in the transformed image which pixel it corresponds int the original image)
    # Thus we don't need inverse
    inv_params = parameters
    p_transf_indices = translation_then_axialrotation(indices,inv_params)
    new_img = canvas_indices_2_trans_img(mask,trans_canvas,p_transf_indices,indices)

    return new_img


def search_best_optimizer(opt_methods: list, ct_img: np.ndarray, phantom_img: np.ndarray):
    for m in opt_methods:
        start_time = time.time()
        result = coregister_imgs(phantom_img,ct_img,opt_method=m)
        elapsed_time = time.time() - start_time
        new_img = apply_transformation(ct_img,phantom_img,result.x,slice_=ct_img.shape[0] // 2)
        print("Method: ", m, "\tMSE: ",np.mean(squared_errors(phantom_img,new_img)),"\tTime: ",elapsed_time)
        print("Result: ", result.x)





    





    
