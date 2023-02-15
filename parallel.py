# 头三个一定要按照这个path,因为main是最早被call的，所以最开始就处理好
import tensorrt
from torch2trt import torch2trt
import torch 
# 上面三个不按照这个顺序就会有bug(主要是环境的bug)
from moviepy.editor import *
import os, shutil, time, math
from main import process_video
from multiprocessing import Process



def sec2foramt(time):
    # Transform second to the format desired
    time = int(time)
    sec = str(time%60)
    sec = "0"*(2-len(sec)) + sec

    time = time//60
    minute = str(time%60)
    minute = "0"*(2-len(minute)) + minute

    time = time//60
    hour = str(time%60)
    hour = "0"*(2-len(hour)) + hour

    format = hour + ":" + minute + ":" + sec
    return format


def check_repeat_file(output_dir):
    if os.path.exists("tmp/"):
        shutil.rmtree("tmp/")
    os.mkdir("tmp/")

    # to avoid annoying Yes or No on cmd by FFMPEG
    target_files = []
    target_files.append(output_dir)

    # Remove unnecessary files
    for file in target_files:
        if os.path.isfile(file):
            os.remove(file)


def split_video(input_file, parallel_num):
    clip = VideoFileClip(input_file)
    divide_time = math.ceil(clip.duration // parallel_num) + 1

    # TODO: 直接拆分audio出来，这样子就不会出现中途有卡壳的情况
    # Split audio
    audio_split_cmd = "ffmpeg -i " + input_file +  " -map 0:a -c copy tmp/output_audio.m4a"
    os.system(audio_split_cmd)

    # Divide videos to segments
    ffmpeg_divide_cmd = "ffmpeg -i  " + input_file +  " -f segment -an -codec copy -loglevel quiet -segment_time " + str(divide_time) + " -reset_timestamps 1 tmp/part%01d.mp4"
    os.system(ffmpeg_divide_cmd)
    
    # handle config setting
    configs = []
    for i in range(parallel_num):
        config = {"inp_path": "tmp/part" + str(i) +".mp4", 
                    "opt_path": "tmp/part" + str(i) +"_res.mp4"}

        configs.append(config)
        

    return configs


def combine_video(target_output, parallel_num, has_subtitle = False):
    # write necessary ffmpeg file
    file = open("tmp/target.txt", "a")
    for i in range(parallel_num):
        file.write("file part"+str(i)+"_res.mp4\n")
    file.close()

    additional_cmd = " -i tmp/output_audio.m4a -c:a aac -strict experimental "
    second_adidional = " "
    if has_subtitle:
        # ffmpeg_combine_cmd = "ffmpeg -f concat -i tmp/target.txt -i tmp/subtitle.srt -loglevel quiet -c copy -c:s mov_text " + target_output
        additional_cmd += " -i tmp/subtitle.srt -c copy -c:s mov_text " # move -c copy bevore -c:s
    else:
        second_adidional = " -c copy "

    ffmpeg_combine_cmd = "ffmpeg -f concat -i tmp/target.txt " + additional_cmd + " -loglevel quiet " + second_adidional +  target_output
    os.system(ffmpeg_combine_cmd)

def check_existence(file):
    if not os.path.isfile(file):
        print("This " + file + " file doesn't exist!")
        os._exit(0)

def extract_subtitle(dir):
    ffmpeg_extract_subtitle_cmd = "ffmpeg -i " + dir + " -map 0:s:0 tmp/subtitle.srt"
    os.system(ffmpeg_extract_subtitle_cmd)


def parallel_process(input_dir, output_dir, args=None, parallel_num = 2):
    print(parallel_num)
    
    check_existence(input_dir)
    check_repeat_file(output_dir)

    has_subtitle = False
    if args:
        if args.extract_subtitle:
            has_subtitle = True
            extract_subtitle(input_dir)

    configs = split_video(input_dir, parallel_num)


    ######################### Double Process ############################
    Processes = []
    for i in range(parallel_num):
        p1 = Process(target=process_video, args =(configs[i], ))
        p1.start()
        Processes.append(p1)
    print("All Processes Start")

    for process in Processes:
        process.join()
        # process.close()
    print("All Processes End")
    ######################################################################


    # combine video together

    combine_video(output_dir, parallel_num, has_subtitle)




def main():
    start = time.time()
    input_dir = r"C:\Users\hikar\Desktop\shinchan1.mp4"
    output_dir = r"C:\Users\hikar\Desktop\shinchan1_processed.mp4"
    parallel_process(input_dir, output_dir)
    full_time_spent = int(time.time() - start)
    print("Total time spent for this video is %d min %d s" %(full_time_spent//60, full_time_spent%60))

if __name__ == "__main__":
    main()



