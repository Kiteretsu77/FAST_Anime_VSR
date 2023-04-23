import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"          # GPU device for inference


class configuration:
    def __init__(self):
        pass


    ######################################################  Frequently Edited Setting  #################################################### 
    scale = 2  # Supported: 1 || 1.5 || 2  (If it's scale!=2, we shrink to (scale/2) * Width/Height and then do SR upscale 2)
    
    

    # TODO: full_model_num有时候=0的时候，全部nt拉满反而会更加快，主要是full_model_num如果一启动，就会put太多full frame，然后必须要足够多的full_model_num来保证同一时间内帧得到处理了，不然就阻塞变慢了
    # Solution: 想办法拉高full frame门槛或者，当数量达到一定queue比例的时候，不要成为full frame（就crop处理），就是要一种dynamic根据queue量来塞的方案
    process_num = 1          # This is a Process number
    full_model_num = 3       # Full frame thread instance number

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

    target_saved_portion = 0.2  #相对于30fps的，如果更加低的fps，应该等比例下降,这也只是个参考值而已，会努力adjust到这个范围，但是最多就0.08-0.7还是保证了performance的
    Queue_hyper_param = 700     #The larger the more queue size allowed and the more cache it will have (higher memory cost, less sleep)

    # 一般来说如果最后的portion是到target_saved_portion，说明其实这个视频原本能压缩的frame绝对多于这个target_saved_portion，因为最高mse就0.7
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