import os
import numpy as np
import torch


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
    return (np.transpose(tensor.squeeze().cpu().numpy(), (1, 2, 0)))    # tensor is already multiplied by 255

def np2tensor(np_frame, pro):
    if pro:
        return torch.from_numpy(np.transpose(np_frame, (2, 0, 1))).unsqueeze(0).cuda().float() / (255 / 0.7) + 0.15
    else:
        return torch.from_numpy(np.transpose(np_frame, (2, 0, 1))).unsqueeze(0).cuda().float() / 255.0
    
