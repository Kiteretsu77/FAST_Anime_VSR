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
1. Nvidia GPU with Cuda (Tested: 3060Ti, 3090Ti, 4090)


# Installation (**Linux - Ubuntu**)：
1. Install CUDA  (The document says that now Cudnn is optional, and I tested that without Cudnn it is still runnable)
2. Install tensorrt (Linux is much simpler by the command below) 
    ```bash
        pip install tensorrt
    ```
&emsp;&emsp; PS: The document says: the Python Package Index installation does not and only supports CUDA 12.x in this release.\
&emsp;&emsp;&emsp;&emsp;For more details, please read https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-pip\
   
3. Install torch2trt (Don't directly use pip isntall torch2trt)\
    &emsp;&emsp; Follow https://nvidia-ai-iot.github.io/torch2trt/v0.2.0/getting_started.html   (I install **Without** plugins)\
    &emsp;&emsp; After installing it, you you should be able to run "import torch2trt" in python (**start a new window** to run this)
4. Install basic lib for python 
    ```bash
        pip install -r requirements.txt
    ```


# Installation (**Windows**)：
1. Install CUDA (Mine: 11.7)
2. Install Cudnn (8.6, I strongly recommend this version!)
3. Install tensorrt (Don't directly use python install) \
    Please strictly Follow https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-zip \
    After finishing these steps, you should be able to "import tensorrt" in python (starts a new window to run this)\
    Some reminder:\
        &emsp;&emsp;Don't forget adding PATH to environmental varialbe based on their requirement.\
        &emsp;&emsp;Step6 in the document is not very needed.

4. install torch2trt (Don't directly use pip isntall torch2trt)\
    &emsp;&emsp; Follow https://nvidia-ai-iot.github.io/torch2trt/v0.2.0/getting_started.html   (I install **Without** plugins)\
    &emsp;&emsp; After installing it, you you should be able to run "import torch2trt" in python (**start a new window** to run this)

5. Install basic lib for python 
    ```bash
        pip install -r requirements.txt
    ```





# Run (Inference):
1. adjust **config.py** to setup your setting. Usually, just editing **Frequently Edited Setting** part is enough. Plaese follow instruction there.\
    &emsp;&emsp;e.g. edit config.py, especially: process_num, full_model_num, nt  (try to set them based on your gpu computation power)
1. Run 
   ```bash
        python main.py
   ```
   PS:\
   &emsp;&emsp;&emsp;1. The **original cunet weight** should be automatically downloaded and **tensorrt transformed weight** should be generated automatically based on input height and weight. \
   &emsp;&emsp;&emsp;2. Usually, if this is the first time you transform to a tensorrt weight, it may need to wait for a while for the program to generate tensorrt version weight. \
   &emsp;&emsp;&emsp;3. The input <span style="color:red"> can be a single video input or a folder </span> with a bunch of videos (input format can be various as long as they are supported by ffmpeg); The output is **mp4** format. \
   &emsp;&emsp;&emsp;4. If the input source has any <span style="color:red">external subtitle</span>, it will also be extracted automtically and sealed back to the processed video at the end.




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

