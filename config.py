import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"    # GPU device for inference


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

    ######################################################  模型通用设置  ####################################################
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

    ######################################################  MSE设置  ########################################################
    momentum = 4  # 暂时用3/4都行吧
    MSE_range = 0.2  # 现在显卡上去了，这个稍微小一点都行了
    Max_Same_Frame = 40  # 基本上50就行了,但是为了一些edge case出问题，建议写大点也行(之前用的是75)
    mse_learning_rate = 0.005 #应该来说是越往小的走越精细，越往大的走越粗糙
    target_saved_portion = 0.2  #相对于30fps的，如果更加低的fps，应该等比例下降,这也只是个参考值而已，会努力adjust到这个范围，但是最多就0.08-0.7还是保证了performance的
    # 一般来说如果最后的portion是到target_saved_portion，说明其实这个视频原本能压缩的frame绝对多于这个target_saved_portion，因为最高mse就0.7
    ########################################################################################################################


    #################################################### 超分视频设置 ########################################################
    # device 真正的定义在最开头的os CUDA_VISIBLE_DEVICES 中、
    device="cuda"
    n_gpu = 1           # currently only 1 gpu and 1 device

    inp_path = r"C:\Users\HikariDawn\Desktop\video\pokemon.mp4"      # file directory
    opt_path = r"C:\Users\HikariDawn\Desktop\pokemon_processed.mp4" #proceesed video store directory
    
    Queue_hyper_param = 700  #500还是需要的（300也行）
    ########################################################################################################################


    #################################################  多线程和编解码设置  ####################################################
    # p_sleep = (0.005, 0.012) decode_sleep = 0.001
    p_sleep = (0.005, 0.015) # 这个多线程的时候用的
    decode_sleep = 0.001 # 这个是每次视频decode的时候用的, 原本是0.001


    #视频编码参数; 通俗来讲，crf变低=高码率高质量，slower=低编码速度高质量+更吃CPU，CPU不够应该调低级别，比如slow，medium，fast，faster
    encode_params = ['-crf', '21', '-preset', 'medium']
    ########################################################################################################################