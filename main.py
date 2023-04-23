# 头三个一定要按照这个path,因为main是最早被call的，所以最开始就处理好
import tensorrt
from torch2trt import torch2trt
import torch 
# 上面三个不按照这个顺序就会有bug(主要是环境的bug)

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
        # whole video process
        print(f"We are going to process all videos in {configuration.inp_path}")
        mass_process(configuration.inp_path, configuration.opt_path)

    elif os.path.exists(configuration.inp_path):
        # single video process
        if os.path.isdir(configuration.opt_path):
            print("The output folder is a folder. This is an error")
            os._exit(0)

        print(f"We are going to process single videos located at {configuration.inp_path}")
        parallel_process()

    else:
        print("We didn't find such location exists!")


    


if __name__ == "__main__":
    main()