# # The first three lib import is needed to be in the following order, else there is a bug of dependency appear
# import tensorrt
# from torch2trt import torch2trt
# import torch 
# ##########################################

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"          # GPU device for inference


class configuration:
    def __init__(self):
        pass

    
    ######################################################  Frequently Edited Setting  #################################################### 
    # Fundamental setting
    # TODO: the rescale mechanism for Real-CUGAN and Real-ESRGAN need more consideration. Currently, what I preset is a rough format.
    use_rescale = False                    # For Real-CUGAN, If its scale != 2, we shrink to (scale/2) * Width/Height and then do SR upscale 2
                                           # For Real-ESRGAN, If its scale != 4, we shrink to (scale/4) * Width/Height and then do SR upscale 4
    scale = 4                              # Real-CUGAN Supported: 2  &&  Real-ESRGAN Supported: 4 
    model_name = "Real-ESRGAN"              # Supported: "Real-CUGAN" || "Real-ESRGAN" (which only support 4x now)
    inp_path = "../videos/crop.mp4"                 # Intput path (can be a single video file or a folder directory with videos)
    opt_path = "../videos/crop_processed.mp4"       # Output path after processing video/s of inp_path
    

    # Auxiliary setting
    decode_fps = 24             # FPS you want the input source be decoded from; If = -1, use original FPS value; I recommend use 24 FPS because Anime are maked from 24 FPS. Thus, some 30 or more FPS anime video is falsely interpolated with unnecessary frames from my perspective. 
    use_tensorrt = True        # tensorrt increase speed a lot; so, it is highly recommended to install it


    # Multithread and Multiprocessing setting 
    process_num = 3             # This is the fully parallel Process number
    full_model_num = 2          # Full frame thread instance number
    nt = 2                      # Partition frame (1/3 part of a frame) instance number 

    # Reference for my 3090Ti setting (almost full power)
    # Input Resolution: process_num x (full_model_num + nt)
    # 720P: 3 x (2 + 2)
    # 540P: 3 x (3 + 2)
    # 480P: 3 x (3 + 3)
    ######################################################################################################################################


    ###########################################  General Details Setting  ################################################################
    adjust = 6
    left_mid_right_diff = [2, -2, 2] # Generally speaking, this is not needed to modify

    # Architecture name
    _architecture_dict = {"Real-CUGAN": "cunet", 
                         "Real-ESRGAN": "rrdb"}
    architecture_name = _architecture_dict[model_name]

    ######################################################################################################################################
    

    ########################################  Redundancy Acceleration Setting  ###########################################################
    # This part is used for redundancy acceleration
    MSE_range = 0.2                 # How much Mean Square Error difference between 2 frames you can tolerate (I choose 0.2) (The smaller it is, the better quality it will have)
    Max_Same_Frame = 40             # how many frames/sub-farmes at most we can jump (40-70 is ok)

    momentum = 4                    # choose 3/4 
    mse_learning_rate = 0.005       # Momentum learning rate (the smaller the better visual qualitydetails)

    # target_saved_portion = 0.2      #相对于30fps的，如果更加低的fps，应该等比例下降,这也只是个参考值而已，会努力adjust到这个范围，但是最多就0.08-0.7还是保证了performance的
    Queue_hyper_param = 700         #The larger the more queue size allowed and the more cache it will have (higher memory cost, less sleep)

    ######################################################################################################################################


    #########################################  Multi-threading and Encoding Setting ######################################################
    # Original Setting: p_sleep = (0.005, 0.012) decode_sleep = 0.001
    p_sleep = (0.005, 0.015)    # Used in Multi-Threading sleep time (empirical value)
    decode_sleep = 0.001        # Used in Video decode


    # Several recommended options for crf and preset:
    #   High Qulity:                                    ['-crf', '19', '-preset', 'slow']
    #   Balanced:                                       ['-crf', '23', '-preset', 'medium']
    #   Lower Quality but Smaller size and Faster:      ['-crf', '28', '-preset', 'fast'] 

    # If you want to save more bits (lower bitrate and lower bit/pixel):
    #   You can use HEVC(H.265) as the encoder by appending ["-c:v", "libx265"], but the whole processing speed will be lower due to the increased complexity

    encode_params = ['-crf', '23', '-preset', 'medium', "-tune", "animation", "-c:v", "libx264"]        
    ######################################################################################################################################


    # TensorRT Weight Generator needed info
    sample_img_dir = "tensorrt_weight_generator/full_sample.png"
    full_croppped_img_dir = "tensorrt_weight_generator/full_croppped_img.png"
    partition_frame_dir = "tensorrt_weight_generator/partition_cropped_img.png"
    weights_dir = "weights/"

    model_full_name = ""
    model_partition_name = ""