# 头三个一定要按照这个path,因为main是最早被call的，所以最开始就处理好
import tensorrt
from torch2trt import torch2trt
import torch 
# 上面三个不按照这个顺序就会有bug(主要是环境的bug)

import os, sys, collections
import shutil
from moviepy.editor import VideoFileClip
from inference import VideoUpScaler
from pathlib import Path
from config import configuration

# import functions from other file
from weight_generation.weight_generator import generate_weight
#  unet_base_name, unet_full_name, momentum, MSE_range, Max_Same_Frame, \
#                     mse_learning_rate, target_saved_portion, device, inp_path, opt_path, nt, full_model_num, n_gpu, Queue_hyper_param, \
#                     p_sleep, decode_sleep, encode_params


def check_existence(file_dir):
    my_file = Path(file_dir)
    if not my_file.is_file():
        print("P:No such file " + file_dir + " exists!")
        os._exit(0)


def weight_justify(config):
    # Check if needed weight is here. If it is, just edit the config

    # find all supported resolution weight
    supported_res = collections.defaultdict(list)
    for weight_name in os.listdir('weights/'):
        if weight_name == "cunet_weight.pth":
            continue
        infos = weight_name.split('_')
        resolution = infos[4]
        width, height = resolution.split('X')
        supported_res[int(width)].append(int(height))
    print("supported resolution is ", supported_res)


    # check if it is existed in supported_res
    video = VideoFileClip(config.inp_path)
    w, h = video.w, video.h
    if config.scale != 2:
        print("shrink target video size by half and then upscale 2")
        w = int(w * (config.scale/2))
        h = int(h * (config.scale/2))


    partition_height = (h//3) + config.adjust + abs(config.left_mid_right_diff[0])
    if w not in supported_res or h not in supported_res[w] or partition_height not in supported_res[w]:
        print("No such orginal resolution (" + str(w) + "X" + str(h) +") weight supported in current folder!")
        print("We are going to generate the weight!!!")

        # Call weight generator
        assert(h<=1080 and w<=1920)
        generate_weight(h, w)

        print("Finish generating the weight!!!")


        # os._exit(0)
    print("This resolution " + str(w) + "X" + str(h) +" is supported in weights available!")

    
    # edit the unet base name for existed weight
    config.unet_full_name = str(w) + "X" + str(h)
    config.unet_partition_name = str(w) + "X" + str(partition_height)
    

    print(config.unet_full_name, config.unet_partition_name)



def config_preprocess(params, config):
    if params != None:
        for param in params:
            if hasattr(config, param):
                setattr(config, param, params[param])
                print("Set new attr for " + param + " to be " + str(getattr(config, param)))

    # check existence of input
    check_existence(config.inp_path)

    weight_justify(config)



def process_video(params = None):
    root_path = os.path.abspath('.')
    sys.path.append(root_path)

    # Preprocess to edit params to the newest version we need
    config_preprocess(params, configuration)


    # TODO: 我觉得这里应该直接读取video height和width然后直接选择模型，不然每次自己手动很麻烦
    video_upscaler = VideoUpScaler(configuration)

    print("="*100)
    print("Current Processing file is ", configuration.inp_path)
    report = video_upscaler(configuration.inp_path, configuration.opt_path)


    print("All Done for video " + configuration.inp_path + " !")
    os._exit(0)


def folder_prepare():
    if os.path.exists("tmp/"):
        shutil.rmtree("tmp/")
    os.mkdir("tmp/")


def main():
    folder_prepare()
    
    process_video()


if __name__ == "__main__":
    main()