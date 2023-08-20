import os, sys

# Import files from the local folder
root_path_ = os.path.abspath('.')
sys.path.append(root_path_)
from config import configuration


def get_partition_height(full_size_height):
    ''' Get the partition height
    Args:
        full_size_height (int):     The input full size image height
    Returns:
        partition_height (int):     The partition height
    '''
    
    partition_height = (full_size_height//3) + (configuration.pixel_padding//3) + configuration.pixel_padding

    return partition_height



def crop4partition(img, position = 3):
    ''' Crop img to THREE parition based on the setting from configuration
    Args:
        img (numpy):        The input full size image
        position (int):     An optional position index for which crop we need to return
    Returns:
        crop1, crop2, crop3 (numpy):   The cropped image
    '''

    # Preparation
    height, width, _ = img.shape
    crop_base = height // 3  # 160 for 480p
    side_extra_padding = configuration.pixel_padding // 3
    pixel_padding = configuration.pixel_padding
 

    # Crop the images with pixel padding ad adjust
    if position == 0:
        crop1 = img[:crop_base + side_extra_padding+pixel_padding, :, :] # :168
        return crop1

    elif position == 1:
        crop2 = img[crop_base+side_extra_padding-pixel_padding : 2*crop_base-side_extra_padding+pixel_padding, :, :] # 156:324
        return crop2

    elif position == 2:
        crop3 = img[2*crop_base-side_extra_padding-pixel_padding:, :, :] # 312:480
        return crop3

    elif position == 3:
        crop1 = img[:crop_base + side_extra_padding+pixel_padding, :, :] # :168
        crop2 = img[crop_base+side_extra_padding-pixel_padding : 2*crop_base-side_extra_padding+pixel_padding, :, :] # 156:324
        crop3 = img[2*crop_base-side_extra_padding-pixel_padding:, :, :] # 312:480
        return (crop1, crop2, crop3)
    
    else:
        raise NotImplementedError("We didn't find such position of the frame ")


    