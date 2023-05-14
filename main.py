# The first three lib import is needed to be in the following order, else there is a bug of dependency appear
import tensorrt
from torch2trt import torch2trt
import torch 

import os, sys, collections
import shutil
from config import configuration

# import function from other files
from single_process import parallel_process
from mass_production import mass_process



def folder_prepare():
    if os.path.exists("tmp/"):
        shutil.rmtree("tmp/")
    os.mkdir("tmp/")

    if not os.path.exists("weights/"):
        os.mkdir("weights/")



def main():
    folder_prepare()

    if os.path.isdir(configuration.inp_path):
        # whole video folder process
        if not os.path.isdir(configuration.opt_path):
            print("The output folder is not an directory when the input folder is a folder")
            os._exit(0)

        print(f"We are going to process all videos in {configuration.inp_path}")
        mass_process(configuration.inp_path, configuration.opt_path)

    elif os.path.exists(configuration.inp_path):
        # single video process
        if os.path.isdir(configuration.opt_path):
            print("The output folder is a folder. This is an error")
            os._exit(0)

        print(f"We are going to process single videos located at {configuration.inp_path}")
        parallel_process(configuration.inp_path, configuration.opt_path, parallel_num=configuration.process_num)

    else:
        print("We didn't find such location exists!")


    


if __name__ == "__main__":
    main()