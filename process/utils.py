import os
import numpy as np
import torch
from torchvision.transforms import ToTensor
from torchvision.utils import save_image



def check_input_support(file):
    lists = file.split('.')
    format = lists[-1]

    accepted_format = ["mp4", "mkv", "mov"]
    if format not in accepted_format:
        print("This format is not supported!")
        os._exit(0)

    return format


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


def tensor2np(tensor):
    return np.transpose(tensor.squeeze().cpu().detach().numpy(), (1, 2, 0))    # tensor is already multiplied by 255

def np2tensor(np_frame, pro):
    tensor = ToTensor()(np_frame).unsqueeze(0).cuda()
    if pro:
        return tensor * 0.7 + 0.15 # Was / (255 / 0.7) + 0.15
    else:
        return tensor
    
