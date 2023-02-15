# AVSRR (Anime Video Super Resolution and Restoration)
This is repositary based on Real-CuGAN (https://github.com/bilibili/ailab/tree/main/Real-CUGAN). 
I implement it in TensorRT version and utilized a frame division algorithm to accelerate it (with a momentum mechanism). Plus, use FFMPEG to decode a smaller FPS for faster processing. Plus, multiprocessing and multithreading.
Feel free to look at this document (https://docs.google.com/presentation/d/1Gxux9MdWxwpnT4nDZln8Ip_MeqalrkBesX34FVupm2A/edit#slide=id.p) for the implementation and algorithm I have used


# Install (Windows)：
1. install cuda (11.7, Mine)
2. install cudnn (8.6, strongly recommend this version!)
3. install tensorrt (Don't directly use python install)
    Please strictly Follow https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-zip
    After finishing these steps, you should be able to "import tensorrt" in python (start a new window after finishing installation to run this)
    Some reminder:
        Don't forget add PATH to environmental varialbe based on its requirement
        Step6 in the document is not very needed

4. install torch2trt
    Follow https://nvidia-ai-iot.github.io/torch2trt/v0.2.0/getting_started.html   (I install Without plugins)
    after installing it, you you should be able to run "import torch2trt" in python (start a new window after finishing installation to run this)

5. pip install -r requirements.txt


# Run:
1. download cunet weight (https://github.com/bilibili/ailab/blob/main/Real-CUGAN/Changelog_CN.md) and name it as "cunet_weight.pth" and put it under the folder "weights/" (you will need to first make the directory "weights")
2. generate weights first by edit your desired Low Resolution input size (lr_h, lr_width) in weight_generation/weight_generator.py in main()
3. run "python weight_generation/weight_generator.py" (Currently no argument is needed)
4. edit config.py, especially: process_num, full_model_num, nt  (try to set them based on your gpu power)
5. run "python mass_production.py" to process all videos inside a folder (needed to edit input_dir && store_dir)
   run "python main.py" to process just one single file (edit input and output directory in config.py by inp_path && store_dir) [This mode doesn't use any multiprocessing, so it's much slower than mass_production.py]

    


# Future Works:
1. use just one file to run any cases (either a folder or just one single video)
2. directly generate weights from main.py and mass_production.py while running for the first time


# Disclaimer:
1. the sample image under weight_generation is just for faster implementation, I don't have copyright for that one. All rights are reserved to their original owners.
2. My code is edited from Real-CUGAN github repository (https://github.com/bilibili/ailab/tree/main/Real-CUGAN)
3. Provide all repository in English.
