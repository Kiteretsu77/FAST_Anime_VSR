# FAST Anime VSRR (Video Super-Resolution and Restoration)
This repository is dedicated to enhancing the Super-Resolution (SR) inference process for Anime videos by fully harnessing the potential of your GPU. It is built upon the foundations of Real-CuGAN (https://github.com/bilibili/ailab/blob/main/Real-CUGAN/README_EN.md) and Real-ESRGAN (https://github.com/xinntao/Real-ESRGAN). 

I've implemented the SR process using **TensorRT**, incorporating a custom frame division algorithm designed to accelerate it. This algorithm includes a video redundancy jump mechanism, akin to video compression Inter-Prediction, and a momentum mechanism.

Additionally, I've employed FFMPEG to decode the video at a reduced frames-per-second (FPS) rate, facilitating faster processing with an almost imperceptible drop in quality. To further optimize performance, I've utilized both multiprocessing and multithreading techniques to fully utilize all available computational resources.

For a more detailed understanding of the implementation and algorithms used, I invite you to refer to this presentation slide: https://docs.google.com/presentation/d/1Gxux9MdWxwpnT4nDZln8Ip_MeqalrkBesX34FVupm2A/edit#slide=id.p.

In my **3060Ti** Desktop version, it can <span style="color:red">**process in Real-Time on 480P Anime videos input (Real-CUGAN)**</span>, which means that **as soon as you finish watching one Anime video, the second Anime Super-Resolution (SR) video is already processed and ready for you to continue watching with just a simple click.**.

Currently, this repository supports **Real-CUGAN** (official) and a **shallow Real-ESRGAN** (6 blocks Anime Image version RRDB-Net provided by Real-ESRGAN). 
<!-- The reason why I trained a model myself is because the original 23 blocks Real-ESRGAN is too big for Anime video and thus their inference speed is extremely slow. Based on my experiment, a 7 blocks RRDB can restore and super-resolve well on Anime videos, which only increase ~25% time than Real-CUGAN model. -->


&emsp;&emsp;\
My ultimate goal is to directly utilize decode information in Video Codec as in this paper (https://arxiv.org/abs/1603.08968), so I use the word "**FAST**" at the beginning. **Though this repository can already process in real-time, this repository will be continuously maintained and developed.**


**If you like this repository, you can give me a star (if you are willing). Feel free to report any problem to me.**
&emsp;&emsp; \
&emsp;&emsp; 



# Visual Improvement (Real-CUGAN)
**Before**:\
![compare1](figures/before.png)

**After 2X scaling**:\
![compare2](figures/processed.png)
&emsp;&emsp; \
&emsp;&emsp; 


# Model supported now:
1. **Real-CUGAN**:   The original model weight provided by BiliBili (from https://github.com/bilibili/ailab/tree/main)
2. **Real-ESRGAN**:  Using Anime version RRDB with 6 Blocks (full model has 23 blocks) (from https://github.com/xinntao/Real-ESRGAN/blob/master/docs/model_zoo.md#for-anime-images--illustrations)
3. **VCISR**:        A model I trained with my upcoming paper methods using Anime training datasets (https://github.com/Kiteretsu77/VCISR-official)


# Supported Devices and Python Version:
1. Nvidia GPU with Cuda (Tested: 2060 Super, 3060Ti, 3090Ti, 4090)
2. Tested on Python 3.10
&emsp;&emsp; \
&emsp;&emsp; 



# Installation (**Linux - Ubuntu**)：
Skip step 3 and 4 if you don't want tensorrt, but they can increase the speed a lot & save a lot of GPU memory.
1. Install CUDA. The following is how I install:
    * My Nvidia Driver in Ubuntu is installed by **Software & Updates** of Ubuntu (Nvidia server driver 525), and the cuda version in nvidia-smi is 12.0 in default, which is the driver API.
    * Next, I install cuda from the official website (https://developer.nvidia.com/cuda-12-0-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local). I install 12.0 version because it's said that the runtime API should be older than driver API cuda (12.0 in nvidia-smi). I use **runfile(local)** to install because that is the easiest option.\
    **During the installation, Leave driver installation ([] Driver [] 525.65.01) blank**. Avoid dual driver installation.
    * After finishing the cuda installation, we need to add their path to the environment
        ```bash
            gedit ~/.bashrc
            // Add the following two at the end of the popped up file (The path may be different, please double check)
            export PATH=/usr/local/cuda-12.0/bin${PATH:+:${PATH}}
            export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
            // Save the file and execute the following in the terminal
            source ~/.bashrc
        ```
    * You should be capable to run "nvcc --version" to know that cuda is fully installed.
   
2. Install CuDNN. The following is how I install:
    * I downloaded it from the official website (https://developer.nvidia.com/rdp/cudnn-download). Usually, **Tar** version (Linux x86_64 (Tar)) is preferred.
    * Then, give the permisions by the following :
        ```bash
            cd /usr/local/cuda
            sudo chmod 666 include
        ```
    * Follow https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installlinux-tar to decompress folder and copy file to the desired location

3. Install tensorrt
   * Download tensorrt 8.6 from https://developer.nvidia.com/nvidia-tensorrt-8x-download (**12.0 Tar pacakge is preferred**)
   * Follow https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-tar to install several wheels.\
        Step4 of this website is like below (**Don't forget to replace YOUR_USERNAME to your home address**): 
        ```bash
            gedit ~/.bashrc
            // Add the following at the end of the popped up file (The path may be different, please double check)
            export LD_LIBRARY_PATH=/home/YOUR_USERNAME/TensorRT-8.6.1.6/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
            // Save the file and execute the following in the terminal
            source ~/.bashrc
        ```
    * After finishing these steps, you should be able to "import tensorrt" in python (starts a new window to run this)
   
4. Install torch2trt (<span style="color:red">**Don't directly use pip install torch2trt**</span>)
    * Follow https://nvidia-ai-iot.github.io/torch2trt/v0.4.0/getting_started.html   (I install **WITHOUT** plugins)\
    * After installing it, you should be able to run "import torch2trt" in python (**start a new window** to run this)
  
5. Install basic libraries for python 
    ```bash
        pip install -r requirements.txt
    ```
    * Install pytorch: please go to https://pytorch.org/get-started/locally/ to install separately based on your needs
&emsp;&emsp; 



# Installation (**Windows**)：
Skip step 3 and 4 if you don't want tensorrt, but they can increase the speed a lot & save a lot of GPU memory.
1. Install CUDA (e.g. https://developer.nvidia.com/cuda-downloads?)
2. Install Cudnn (move bin\ & include\ & lib\ so far is enough)
3. Install tensorrt (<span style="color:red">**Don't directly use python install**</span>) 
    * Please strictly Follow https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-zip 
    * After finishing these steps, you should be able to "import tensorrt" in python (starts a new window to run this)
    * Don't forget to add PATH to environmental variables based on their requirement.
    * Step 6 (Visual Studio Verification) in the document is not very needed.
  
4. Install torch2trt (<span style="color:red">**Don't directly use pip install torch2trt**</span>)
    * Follow https://nvidia-ai-iot.github.io/torch2trt/v0.4.0/getting_started.html   (I install **Without** plugins)
    * In some rare cases, you may neeed to "pip install packaging" before executing its commands.
    * After installing it, you should be able to run "import torch2trt" in python (**start a new window** to run this)

5. Install basic libraries for python 
    ```bash
        pip install -r requirements.txt
    ```
    * For Pytorch, please go to https://pytorch.org/get-started/locally/ to install seperately for your need
&emsp;&emsp; 




# Run (Inference):
1. Adjust **config.py** to setup your setting. Usually, just editing **Frequently Edited Setting** part is enough. Please follow the instructions there.
    * Edit **process_num**, **full_model_num**, **nt** to match your GPU's computation power.
    * The input (inp_path) can be **a single video input** or **a folder with a bunch of videos** (video format can be various as long as they are supported by ffmpeg); The output is **mp4** format in default. 

2. Run 
   ```bash
        python main.py
   ```
    * The <span style="color:red">**original cunet weight** should be automatically downloaded </span> and **tensorrt transformed weight** should be generated automatically based on the video input height and weight. 
    * Usually, if this is the first time you transform to a tensorrt weight, it may need to wait for a while for the program to generate tensorrt weight. 
    * If the input source has any **external subtitle**, it will also be extracted automatically and sealed back to the processed video at the end.
&emsp;&emsp; \
&emsp;&emsp; 




# Future Works:
1. Debug use without TensorRT && when full_model_num=0
1. MultiGPU inference support
2. Provide PSNR && Visual Quality report in README.md
3. Provide all repositories in English.
4. Record a video on how to install TensorRT from scratch.
&emsp;&emsp; \
&emsp;&emsp; 


# Disclaimer:
1. The sample image under tensorrt_weight_generator is just for faster implementation, I don't have a copyright for that one. All rights are reserved to their original owners.
1. My code is developed from Real-CUGAN github repository (https://github.com/bilibili/ailab/tree/main/Real-CUGAN)

