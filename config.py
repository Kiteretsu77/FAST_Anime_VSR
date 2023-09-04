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

    ######################################################  Frequently Edited Setting  ########################################################################################
    
    # Updates and Notes:
    #   A new feature of hardware encode ("hevc_nvenc") have been added. 
    #   If you don't feel that your GPU memory and utilization has spare resources, we recommend you to change "hevc_nvenc" to "libx264" for encode_params setting

    ########################################################### Fundamental Setting #######################################################################################   
    model_name = "VCISR"                  # Supported: "Real-CUGAN" || "Real-ESRGAN" || "VCISR"
    inp_path = "../videos/pokemon1.mp4"                 # Intput path (can be a single video file or a folder directory with videos)
    opt_path = "../videos/pokemon1_processed.mp4"       # Output path after processing video/s of inp_path (PS: If inp_path is a folder, opt_path should also be a folder)
    rescale_factor = 1                          # What rescale for the input frames before doing Super-Resolution [Use this way to take less computation for SR model]
                                                # [default 1 means no rescale] We recommend use some value like 0.5, 0.25 to avoid invalid input size in certain minor cases            
    #######################################################################################################################################################################



    # Auxiliary setting
    decode_fps = 24          # FPS you want the input source be decoded from; If = -1, use original FPS value; I recommend use 24 FPS because Anime are maked from 23.98 (~24) FPS. Thus, some 30 or more FPS anime video is falsely interpolated with unnecessary frames from my perspective. 
    use_tensorrt = True      # Tensorrt increase speed a lot; So, it is highly recommended to install it
    use_rename = False       # Sometimes the video that users download may include unsupported characters, so we rename it if this one is True

    # Multithread and Multiprocessing setting 
    process_num = 3          # The numver of times we split the video and process totally in parallel
    full_model_num = 2       # Full frame thread instance number
    nt = 2                   # Partition frame (1/3 part of a frame) instance number 

    # PS:
    #   Reference for my 5600x + 3090Ti setting for Real-CUGAN (almost full power)
    #   **For Real-ESRGAN there is some bugs when nt != 0, I am still analyzing it. To use Real-ESRGAN, we recommend to set nt = 0**
    #   Input Resolution: process_num x (full_model_num + nt)
    # 720P: 3 x (2 + 2)
    # 540P: 3 x (3 + 2)
    # 480P: 3 x (3 + 3)
    ##########################################################################################################################################################################


    ###########################################  General Details Setting  ################################################################
    pixel_padding = 6                                 # This value should be divisible by 3 (and 2 also)  

    # Architecture name or private info
    _architecture_dict = {"Real-CUGAN": "cunet", 
                          "Real-ESRGAN": "rrdb",
                          "VCISR" : "rrdb",
                         }
    architecture_name = _architecture_dict[model_name]
    
    _scale_base_dict = {"Real-CUGAN": 2, 
                        "Real-ESRGAN": 4,
                        "VCISR": 2}
    scale = _scale_base_dict[model_name]   
    scale_base = _scale_base_dict[model_name]
    ######################################################################################################################################
    

    ########################################  Redundancy Acceleration Setting  ###########################################################
    # This part is used for redundancy acceleration
    MSE_range = 0.2                         # How much Mean Square Error difference between 2 frames you can tolerate (I choose 0.2) (The smaller it is, the better quality it will have)
    Max_Same_Frame = 40                     # How many frames/sub-farmes at most we can jump (40-70 is ok)
    momentum_skip_crop_frame_num = 4        # Use 3 || 4 

    target_saved_portion = 0.2      #相对于30fps的，如果更加低的fps，应该等比例下降,这也只是个参考值而已，会努力adjust到这个范围，但是最多就0.08-0.7还是保证了performance的
    Queue_hyper_param = 700         #The larger the more queue size allowed and the more cache it will have (higher memory cost, less sleep)

    ######################################################################################################################################


    #########################################  Multi-threading and Video Encoding Setting ######################################################
    # Original Setting: p_sleep = (0.005, 0.012) decode_sleep = 0.001
    p_sleep = (0.005, 0.015)    # Used in Multi-Threading sleep time (empirical value)
    decode_sleep = 0.001        # Used in Video decode


    # Several recommended options for crf (higher means lower quality) and preset (faster means lower quality but less time):
    #   High Qulity:                                    ['-crf', '19', '-preset', 'slow']
    #   Balanced:                                       ['-crf', '23', '-preset', 'medium']
    #   Lower Quality but Smaller size and Faster:      ['-crf', '28', '-preset', 'fast'] 

    # Note1: If you feel that your GPU has unused power (+unsued GPU memory) and CPU is almost occupied:
    #   You should USE the DEFAULT ["-c:v", "hevc_nvenc"], this will increase the speed (hardware encode release CPU pressure and accelerate the speed)
    # Note2: If you want to have a lower data size (lower bitrate and lower bits/pixel):
    #   You can use HEVC(H.265) as the encoder by appending ["-c:v", "libx265"], but the whole processing speed will be lower due to the increased complexity

    encode_params = ['-crf', '23', '-preset', 'medium', "-tune", "animation", "-c:v", "hevc_nvenc"]        
    ######################################################################################################################################


    # TensorRT Weight Generator needed info
    sample_img_dir = "tensorrt_weight_generator/full_sample.png"
    full_croppped_img_dir = "tensorrt_weight_generator/full_croppped_img.png"
    partition_frame_dir = "tensorrt_weight_generator/partition_cropped_img.png"
    weights_dir = "weights/"

    model_full_name = ""
    model_partition_name = ""