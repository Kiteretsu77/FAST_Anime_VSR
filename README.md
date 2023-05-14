# FAST Anime VSRR (Video Super Resolution and Restoration)
This is a repositary to acelerate Super Resolution (SR) in Anime video.
It's based on Real-CuGAN (https://github.com/bilibili/ailab/blob/main/Real-CUGAN/README_EN.md). 
I implement it in **TensorRT** version and utilized a frame division algorithm (self-designed) to accelerate it(with a video redunduncy jump mechanism and a momentum mechanism). Plus, I use FFMPEG to decode a smaller FPS for faster processing. Plus, multiprocessing and multithreading.

In my **3060Ti** Desktop version, it can process <span style="color:red">**faster than the Real-Time Anime videos**</span>, which means that **when you finish watching the first Anime video, your second Anime SR video is already processed and you just need to click it to continue watching the next one**.

Feel free to look at this document (https://docs.google.com/presentation/d/1Gxux9MdWxwpnT4nDZln8Ip_MeqalrkBesX34FVupm2A/edit#slide=id.p) for the implementation and algorithm I have used.

My ultimate target is to directly utilize decode information in Video Codec as in this paper (https://arxiv.org/abs/1603.08968), so I use the word "FAST" at the beginning.


**If you like this repository, you can give me a star (if you are willing). Feel free to report any problem to me.**


# Visual Improvement
**Before**:\
![compare1](figures/before.png)

**After 2X scaling**:\
![compare2](figures/processed.png)


# Supported Devices:
1. Nvidia GPU with Cuda

# Installation (Windows)：
1. Install cuda (11.7, Mine)
2. Install cudnn (8.6, strongly recommend this version!)
3. Install tensorrt (Don't directly use python install) \
    Please strictly Follow https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-zip \
    After finishing these steps, you should be able to "import tensorrt" in python (start a new window after finishing installation to run this)\
    Some reminder:\
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Don't forget adding PATH to environmental varialbe based on their requirement.\
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Step6 in the document is not very needed.

4. install torch2trt (Don't directly use pip isntall torch2trt)
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Follow https://nvidia-ai-iot.github.io/torch2trt/v0.2.0/getting_started.html   (I install **Without** plugins)\
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; After installing it, you you should be able to run "import torch2trt" in python (start a new window after finishing installation to run this)

5. pip install -r requirements.txt




# Run (Inference):
<!-- 1. download cunet weight (https://github.com/bilibili/ailab/blob/main/Real-CUGAN/Changelog_CN.md) and name it as "cunet_weight.pth" and put it under the folder "weights/" (you will need to first make the directory "weights") -->
<!-- 1. generate weights first by edit your desired Low Resolution input size (lr_h, lr_width) in weight_generation/weight_generator.py in main() -->
1. adjust "config.py" to setup your setting. Usually, just editing "Frequently Edited Setting" part is enought. Plaese follow instruction there.
    e.g. edit config.py, especially: process_num, full_model_num, nt  (try to set them based on your gpu computation power)
1. Run '''python main.py'''\
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**original cunet weight** should be automatically downloaded and **tensorrt weight transform** should be generated automatically based on input height and weight\
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Usually, if this is the first time you transform to a tensorrt weight, it may need to wait for a while.\
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The input can be a single video input or a folder with a bunch of videos (input format can be various as long as they are supported by ffmpeg); The output is **mp4** format. \
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;If the input source has any **external subtitle**, it will also be extracted automtically and sealed back to the processed video at the end.\




<!-- 1. run "python mass_production.py" to process all videos inside a folder (needed to edit input_dir && store_dir)
   run "python main.py" to process just one single file (edit input and output directory in config.py by inp_path && store_dir) [**This mode doesn't use any multiprocessing**, so it's **much slower than mass_production.py**]
    (Wait me to update parallel.py) -->

# Several things to be aware of:



# Future Works:
1. Provide PSNR && Visual Quality report in README.
1. Provide all repository in English.
1. Setup whole repository in Linux (Ubuntu)
1. Record a video for how to install TensorRT from scratch


# Disclaimer:
1. The sample image under weight_generation is just for faster implementation, I don't have copyright for that one. All rights are reserved to their original owners.
1. My code is developed from Real-CUGAN github repository (https://github.com/bilibili/ailab/tree/main/Real-CUGAN)

