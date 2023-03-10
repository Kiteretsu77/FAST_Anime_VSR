import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"          # GPU device for inference


class configuration:
    def __init__(self):
        pass

    ############################################## Frequently Edited Setting  ##############################################
    scale = 2  # Supported: 1|| 1.5 || 2  (If it's scale=1, we shrink to half height and width of it and upscale 2; If the size is 1.5, shrink by 0.75 of height and width and then upscale)
    

    # Very important hyper parameter
    process_num = 2          # This is a Process number
    full_model_num = 2       # Full frame thread instance number
    nt = 2                   # Partition frame (1in3) instance number 

    # Reference for my 3090Ti setting (almost full power)
    # 720P: 3 x (2 + 2)
    # 540P: 2 x (3 + 2)
    # 480P: 3 x (3 + 3)

    ########################################################################################################################

    ###########################################  General Setting  ##########################################################
    # target_height = 576  # ***important***
    # screen_portion = "5:4" # ***important***
    # unet_name_info = {480: {"4:3": ["640X168", "640X480"], "16:9":["720X168", "720X480"]},
    #                 576: {"5:4": ["720X200", "720x576"]},
    #                 720: {"16:9": ["1280X248", "1280X720"]}}
    # unet_partition_name = unet_name_info[target_height][screen_portion][0]
    # unet_full_name = unet_name_info[target_height][screen_portion][1]

    unet_partition_name = ""
    unet_full_name = ""
    adjust = 6
    left_mid_right_diff = [2, -2, 2]

    ########################################################################################################################

    ########################################  Redundancy Acceleration Setting  #############################################
    # This part is used for redundancy acceleration
    MSE_range = 0.2                 # How much Mean Square Error difference you can tolerate (I choose 0.2) (The smaller it is, the better quality it will have)
    Max_Same_Frame = 40             # how many frames/sub-farmes at most we can jump (40-70 is ok)

    momentum = 4                    # choose 3/4 
    mse_learning_rate = 0.005       # Momentum learning rate (the smaller the better visual qualitydetails)

    target_saved_portion = 0.2  #?????????30fps????????????????????????fps????????????????????????,??????????????????????????????????????????adjust?????????????????????????????????0.08-0.7???????????????performance???
    Queue_hyper_param = 700     #The larger the more queue size allowed and the more cache it will have (higher memory cost, less sleep)

    # ???????????????????????????portion??????target_saved_portion?????????????????????????????????????????????frame??????????????????target_saved_portion???????????????mse???0.7
    ########################################################################################################################


    ########################################### input & output folder setting ##############################################
    # GPU device 
    device="cuda"
    n_gpu = 1           # currently only 1 gpu

    inp_path = r"C:\Users\HikariDawn\Desktop\video\pokemon.mp4"                 # file directory just for running "python main.py"
    opt_path = r"C:\Users\HikariDawn\Desktop\pokemon_processed.mp4"             # proceesed video store directory
    ########################################################################################################################


    #############################################  Multi-threading and Encoding ###########################################
    # p_sleep = (0.005, 0.012) decode_sleep = 0.001
    p_sleep = (0.005, 0.015)    # Used in Multi-Threading sleep time (empirical value)
    decode_sleep = 0.001        # Used in Video decode


    # Several recommended options:
    #   High Qulity and Slow:       ['-crf', '19', '-preset', 'slow']
    #   Balanced:                   ['-crf', '23', '-preset', 'medium']
    #   Lower Quality and False:    ['-crf', '28', '-preset', 'fast'] 
    encode_params = ['-crf', '23', '-preset', 'medium']
    ########################################################################################################################