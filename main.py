import tensorrt
from torch2trt import torch2trt
import torch 

import os, sys
import shutil
import collections
from moviepy.editor import VideoFileClip
# import multiprocessing
# multiprocessing.set_start_method('spawn')

# import from local folder
root_path_ = os.path.abspath('.')
sys.path.append(root_path_)
from config import configuration
from process.single_video import parallel_process
from process.mass_production import mass_process


def folder_prepare():
    if os.path.exists("tmp/"):
        shutil.rmtree("tmp/")
    os.mkdir("tmp/")

    if not os.path.exists(configuration.weights_dir):
        os.mkdir(configuration.weights_dir)
    
    folder_dir = os.path.join(configuration.weights_dir, configuration.model_name)
    if not os.path.exists(folder_dir):
        os.mkdir(folder_dir)
 


def main():
    ''' The main caller of all program, it will distinguish if the input is a folder or a single video and deal with them differently
    '''
    # Prepare folder
    folder_prepare()

    # Convenient attribute here
    input_path  = configuration.inp_path
    output_path = configuration.opt_path


    # Handle a whole directory
    if os.path.isdir(input_path):
        # whole video folder process
        if os.path.exists(output_path) and not os.path.isdir(output_path):
            print("The output folder is not an directory when the input folder is a folder")
            os._exit(0)
        elif not os.path.exists(output_path):
            print("The output directory doesn't exists, we will make one")
            os.mkdir(output_path)       # It is better to ensure here that 

        print(f"We are going to process all videos in {input_path}")
        mass_process(input_path, output_path)


    # Handle a single video
    elif os.path.exists(input_path):
        # single video process
        if os.path.isdir(output_path):
            print("The output folder is a folder. This is an error")
            os._exit(0)

        print(f"We are going to process single videos located at {input_path}")
        parallel_process(input_path, output_path, parallel_num=configuration.process_num)

    else:
        print("We didn't find such location exists!")


    


if __name__ == "__main__":
    main()