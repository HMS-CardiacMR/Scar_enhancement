from __future__ import print_function
import numpy as np
import cv2

def get_masks(cardiac_data, label_data):
    """ Function to extract segmentation masks for each of the tissues of interest.
       Params:
          - cardiac_data: the LGE image of size 256x256
          - label_data: segmentation mask with values [0,1,2,3] of size 256x256
       Returns:
          - mask_blood: mask for the blood pool (label -> 1)
          - mask_myo: mask for the myocardium (label -> 2)
          - mask_scar: mask for the scar (label -> 3)
    """
    mask_blood = label_data == 1.0
    mask_myo = label_data == 2.0
    mask_scar = label_data == 3.0
    return mask_blood, mask_myo, mask_scar

def mri_masked(cardiac_data, label_data):
    """ Function to extract values of each of the tissues of interest.
       Params:
          - cardiac_data: the LGE image of size 256x256
          - label_data: segmentation mask with values [0,1,2,3] of size 256x256
       Returns:
          - cardiac_blood: values of the blood pool 
          - cardiac_myo: values of the myocardium 
          - cardiac_scar: values of the scar 
    """
    mask_blood, mask_myo, mask_scar = get_masks(cardiac_data, label_data)
    cardiac_blood = cardiac_data[mask_blood]
    cardiac_myo = cardiac_data[mask_myo]
    cardiac_scar = cardiac_data[mask_scar]
    return cardiac_blood, cardiac_myo, cardiac_scar

def normalize_image(image, min_value, max_value):
    """ A function to normalizing an image into a specified range [min_value, max_value].
       Params:
          - image: the image to be normalized
          - min_value: minimum value for the normalization
          - max_value: maximum value for the normalization
       Returns:
          - normalized_image: input image normalized into the specified range
    """
    current_min = np.min(image)
    current_max = np.max(image)
    normalized_image = (image - current_min) / (current_max - current_min) * (max_value - min_value) + min_value
    return normalized_image

def adjust_gamma(image, gamma=1.0):
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

def cnr(tissue1, tissue2):
    """ A function calculating the contrast-to-noise ratio (CNR) between two tissues.
       Params:
          - tissue1: selected tissue for the CNR
          - tissue2: the other selected tissue for the CNR
       Returns:
          - CNR of the two inputted tissues
    """
    return np.abs((np.mean(tissue1) - np.mean(tissue2))) / np.std(tissue2)

def tissue_specific_gamma_correction(mr, lbl):
    """ Function generating tissue specific gamma correction with values setup for my specific data.
       Params:
          - mr: the LGE image of size 
          - lbl: segmentation mask with values [0,1,2,3] 
       Returns:
          - mr_edited: gamma corrected image
          - blood: gamma corrected blood pool values
          - myo: gamma corrected healthy myocardium values
          - scar: gamma corrected scar values
    """
    mr_edited = np.array(mr)
    mr_edited = normalize_image(mr_edited, 0, 255) # normalize all images to the same range
    blood_mask, myo_mask, scar_mask = get_masks(mr_edited, lbl)
    blood, myo, scar = mri_masked(mr_edited, lbl) 
    if len(scar) != 0:
        if np.mean(scar) < 60.46:
            scar = adjust_gamma(scar.astype("uint8"), gamma=1.7) # adjust gamma for the values outside the area of interest
        elif np.mean(scar) >= 60.46 and np.mean(scar) < 150:
            scar = adjust_gamma(scar.astype("uint8"), gamma=1.6) # adjust gamma for the values outside the area of interest
        else:
            scar = adjust_gamma(scar.astype("uint8"), gamma=1.5) # adjust gamma for the values outside the area of interest
        mr_edited[scar_mask] = scar.ravel()
    if len(blood) != 0:
        blood_mean = np.mean(blood)
        if blood_mean > 56.47:
            if blood_mean < 86.03:
                blood = adjust_gamma(blood.astype("uint8"), gamma=0.9)
            elif blood_mean < 115.59 and blood_mean >= 86.03:
                blood = adjust_gamma(blood.astype("uint8"), gamma=0.8)
            elif blood_mean >= 115.59 and blood_mean < 130.95:
                blood = adjust_gamma(blood.astype("uint8"), gamma=0.7)
            else:
                blood = adjust_gamma(blood.astype("uint8"), gamma=0.6)
            mr_edited[blood_mask] = blood.ravel()    
    return mr_edited, blood, myo, scar

def extract_roi(slice, slice_lbl):
    """ Function to extract the ROI region, i.e, where the label values are not equal to 0.
       Params:
          - slice: the LGE slice 
          - slice_lbl: segmentation mask 
       Returns:
          - cropped version of the image, including ROI only
    """
    nonzero_pixels = np.argwhere(slice_lbl != 0)
    min_row, min_col = np.min(nonzero_pixels, axis=0)
    max_row, max_col = np.max(nonzero_pixels, axis=0)
    return slice[min_row:max_row, min_col:max_col]
