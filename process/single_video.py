# import tensorrt
# from torch2trt import torch2trt
# import torch 

import os, sys, collections, time
import shutil, math
from moviepy.editor import VideoFileClip
from process.inference import VideoUpScaler
from pathlib import Path
from config import configuration
from multiprocessing import Process


# import from local folder
root_path_ = os.path.abspath('.')
sys.path.append(root_path_)
from process.utils import check_input_support
from tensorrt_weight_generator.weight_generator import generate_weight



def check_existence(file_dir):
    my_file = Path(file_dir)
    if not my_file.is_file():
        print("P:No such file " + file_dir + " exists!")
        os._exit(0)


def config_preprocess(params, config):
    # Here, we should replace configuration.inp_path to be a specific partitioned file we need.
    if params != None:
        for param in params:
            if hasattr(config, param):
                setattr(config, param, params[param])
                print("Set new attr for " + param + " to be " + str(getattr(config, param)))

    # # check existence of input
    # check_existence(config.inp_path)


def check_repeat_file(output_dir):
    if os.path.exists("tmp/"):
        shutil.rmtree("tmp/")
    os.mkdir("tmp/")

    # to avoid annoying Yes or No to delete files on cmd of FFMPEG
    target_files = []
    target_files.append(output_dir)

    # Remove unnecessary files
    for file in target_files:
        if os.path.isfile(file):
            os.remove(file)


def weight_justify(config, video_input_dir):
    ''' Check if our needed weight is inside this folder. If it is, edit config; else, download and generate one
    '''
    

    # Find all supported resolution weight
    supported_res = collections.defaultdict(list)
    for weight_name in os.listdir(os.path.join(configuration.weights_dir, configuration.model_name)):
        if weight_name.find(configuration.architecture_name) != -1:
            # jump pretrained weight pth
            continue
        infos = weight_name.split('_')
        resolution = infos[1]
        width, height = resolution.split('X')
        supported_res[int(width)].append(int(height))
    print("Current supported input resolution for Super-Resolution is ", supported_res)


    # Check if it is existed in supported_res
    video = VideoFileClip(video_input_dir)
    w, h = video.w, video.h
    if config.scale != 2:
        print("Shrink target video size by half and then upscale 2")
        w = int(w * (config.scale/2))
        h = int(h * (config.scale/2))


    # Generate the TensorRT weight if needed
    partition_height = (h//3) + config.adjust + abs(config.left_mid_right_diff[0])
    if w not in supported_res or h not in supported_res[w] or partition_height not in supported_res[w]:
        print("No such orginal resolution (" + str(w) + "X" + str(h) +") weight supported in current folder!")
        print("We are going to generate the weight now!!!")

        # Call TensorRT weight generator
        assert( h <= 1080 and w <= 1920 )
        generate_weight(h, w)               # It will automatically read model name we need

        print("Finish generating the weight!!!")

    print("This resolution " + str(w) + "X" + str(h) +" is supported!")

    
    # Edit the model base name for existed weight (This will be used in inference by directly accessing configuration)
    model_full_name = str(w) + "X" + str(h)
    model_partition_name = str(w) + "X" + str(partition_height)
    # print("The full frame name is {} and partition frame name is {} ".format(configuration.model_full_name, configuration.model_partition_name))


    return (model_full_name, model_partition_name)


def split_video(input_file, parallel_num):
    clip = VideoFileClip(input_file)
    divide_time = math.ceil(clip.duration // parallel_num) + 1

    # TODO: 直接拆分audio出来，这样子就不会出现中途有卡壳的情况
    # Split audio
    audio_split_cmd = "ffmpeg -i " + input_file +  " -map 0:a -c copy tmp/output_audio.m4a"
    os.system(audio_split_cmd)

    # Divide videos to segments
    ffmpeg_divide_cmd = "ffmpeg -i  " + input_file +  " -f segment -an -codec copy -loglevel quiet -segment_time " + str(divide_time) + " -reset_timestamps 1 tmp/part%01d." + configuration.input_video_format
    os.system(ffmpeg_divide_cmd)
    
    # handle config setting
    configs = []
    for i in range(parallel_num):
        config = {"inp_path": "tmp/part" + str(i) +"." + configuration.input_video_format, 
                    "opt_path": "tmp/part" + str(i) +"_res." + configuration.input_video_format}

        configs.append(config)
        

    return configs


def combine_video(target_output, parallel_num):
    # write necessary ffmpeg file
    file = open("tmp/target.txt", "a")
    for i in range(parallel_num):
        file.write("file part"+str(i)+"_res."+ configuration.input_video_format+"\n")
    file.close()

    # If audio exists, we can append them inside the final output video
    additional_cmd = " "
    if os.path.exists("tmp/output_audio.m4a"):
        additional_cmd += " -i tmp/output_audio.m4a -c:a aac -strict experimental "
    

    second_adidional = " "
    if os.path.exists("tmp/subtitle.srt"):
        # If subtitle exists, we can append them inside the final output video
        additional_cmd += " -i tmp/subtitle.srt -c copy -c:s mov_text " # move -c copy bevore -c:s
    else:
        second_adidional = " -c copy "

    ffmpeg_combine_cmd = "ffmpeg -f concat -i tmp/target.txt " + additional_cmd + " -loglevel quiet " + second_adidional +  target_output
    os.system(ffmpeg_combine_cmd)



def extract_subtitle(dir):
    ffmpeg_extract_subtitle_cmd = "ffmpeg -i " + dir + " -map 0:s:0 tmp/subtitle.srt"
    os.system(ffmpeg_extract_subtitle_cmd)
    

def parallel_process(input_path, output_path, parallel_num = 2):
    ''' Split video into several parts and super-resolve each seperately (This function will be called no matter what if the input is a single video or a foldder)
    Args:
        input_path (str): single video path
        output_path (str): output path
        parallel_num (int): how many videos you want to process completely parallelly
    '''
    
    # Check file preparation
    check_existence(input_path)
    check_repeat_file(output_path)
    video_format = check_input_support(input_path)
    configuration.input_video_format = video_format


    # Prepare TensorRT weight (Detect this every time when you are processing a different video)
    model_full_name, model_partition_name = weight_justify(configuration, input_path)
    print("The full frame name is {} and partition frame name is {} ".format(configuration.model_full_name, configuration.model_partition_name))

    # Extract subtitle automatically no matter if it has or not
    extract_subtitle(input_path)


    # Split video to process parallel
    parallel_configs = split_video(input_path, parallel_num)


    ################################### Double Process ################################################
    start_time = time.time()
    Processes = []
    for process_id in range(parallel_num):
        p1 = Process(target=single_process, args =(model_full_name, model_partition_name, parallel_configs[process_id], process_id))
        p1.start()
        Processes.append(p1)
    print("All Processes Start")

    for process in Processes:
        process.join()
        # process.close()
    print("All Processes End")
    full_time_spent = int(time.time() - start_time)
    print("Total time spent for this video is %d min %d s" %(full_time_spent//60, full_time_spent%60))
    ####################################################################################################
    total_duration = VideoFileClip(input_path).duration
    print("The scaling of processing_time/total_video_duration is {} %".format((full_time_spent/total_duration) * 100))


    # combine video together
    combine_video(output_path, parallel_num)



def single_process(model_full_name, model_partition_name, params, process_id):

    # Preprocess to edit params to the newest version we need
    config_preprocess(params, configuration)


    # TODO: 我觉得这里应该直接读取video height和width然后直接选择模型，不然每次自己手动很麻烦
    video_upscaler = VideoUpScaler(configuration, model_full_name, model_partition_name, process_id)
    print("="*100)
    print("Current Processing file is ", configuration.inp_path)
    report = video_upscaler(configuration.inp_path, configuration.opt_path)
    print("Done for video " + configuration.inp_path + " !")
    os._exit(0)