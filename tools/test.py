import sys, os
from moviepy.editor import *
import cv2
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
from torch2trt import TRTModule
import torch
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]="0"


# Test if RealESRGAN can support low resolution (1in3) input
root_path = os.path.abspath('.')
sys.path.append(root_path)
from process.utils import np2tensor, tensor2np
from Real_ESRGAN.rrdb import RRDBNet


height = 248
pretrained = False



# Load model
if pretrained:
    rrdb_model = RRDBNet().cuda()
    model_weight = torch.load("weights/Real-ESRGAN/rrdb_weight.pth")
    model_weight = model_weight['model_state_dict']
else:
    rrdb_model = TRTModule()
    model_weight = torch.load("weights/Real-ESRGAN/trt_1280X"+str(height)+"_float16_weight.pth")


rrdb_model.load_state_dict(model_weight)
for param in rrdb_model.parameters():
    param.grad = None


# Extract frames
objVideoreader = VideoFileClip(filename="/home/hiakaridawn2/Desktop/videos/test.mp4")
for idx, frame in enumerate(objVideoreader.iter_frames()):
    print(idx)
    if idx == 10:
        break

    # crop the frame
    img = frame[:height, :, :]


    img = ToTensor()(img).unsqueeze(0).cuda()
    if not pretrained:
        img = img.half()
    

    # generate the output
    img = rrdb_model(img)

    save_image(img, str(idx) + ".png")


