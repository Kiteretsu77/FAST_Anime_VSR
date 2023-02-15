# 头三个一定要按照这个path
import tensorrt
from torch2trt import torch2trt
import torch 
# 上面三个不按照这个顺序就会有bug(主要是环境的bug)

from torch2trt import TRTModule
from torch import nn as nn
from torch.nn import functional as F
from time import time as ttime
import numpy as np
import os, sys


root_path = os.path.abspath('.')
sys.path.append(root_path)

class UpCunet2x(nn.Module):
    def __init__(self, unet_full_weight_path, device_name, adjust):
        super(UpCunet2x, self).__init__()
        load_start = ttime()


        torch.cuda.empty_cache()
        self.unet_model_full = TRTModule()

        
        self.unet_model_full.load_state_dict(torch.load(unet_full_weight_path))
        # self.unet_model_full.eval().cuda()

        for param in self.unet_model_full.parameters():
            param.grad = None


        self.adjust_double = 2*adjust
        print("torch2trt unet full load+prepare time %.3f s"%(ttime() - load_start))



    def forward(self, x, position):
        x = F.pad(x, (18, 18, 18, 18), 'reflect')  # pad最后一个倒数第二个dim各上下18个（总计36个）

        ######################## Neural Network Process #############################
        unet_full_output = self.unet_model_full(x)
        #############################################################################


        ######################## Afetr Process ######################################

        # 根据各个frame的position（上面，中间，下面，还是全部）来进行拆分adjust
        if position == 0:
            x = unet_full_output[:, :, :-self.adjust_double, :]
        elif position == 1:
            x = unet_full_output[:, :, self.adjust_double:-self.adjust_double, :]
        elif position == 2:
            x = unet_full_output[:, :, self.adjust_double:, :]
        elif position == 3:
            # Full Frame Model
            x = unet_full_output
        else:
            print("Error Type!")



        # TODO: 想办法加check是否是奇数的情况
        # 目前默认是pro mode (pro跟weight有关)
        return ((x - 0.15) * (255/0.7)).round().clamp_(0, 255).byte()

        

    
class RealCuGAN_Scalar(object):
    def __init__(self, unet_full_weight_path, device_name, adjust):
        self.model = UpCunet2x(unet_full_weight_path, device_name, adjust)
        self.inner_times = 0
        self.counter = 0
        self.device_name = device_name

    def __del__(self):
        # if self.counter:
        #     print("Inner time is %.2f s on %d, which is %.5f s per frame"%(self.inner_times, self.counter, self.inner_times/self.counter))
        return

    def tensor2np(self, tensor):
        # 这边看看还有没有什么能够提升的点，耗时实在太长了
        return (np.transpose(tensor.squeeze().cpu().numpy(), (1, 2, 0)))

    def np2tensor(self, np_frame):
        # return torch.from_numpy(np.transpose(np_frame, (2, 0, 1))).unsqueeze(0).to(self.device).float() / 255
        ###### pro mode
        return torch.from_numpy(np.transpose(np_frame, (2, 0, 1))).unsqueeze(0).cuda().half() / (255 / 0.7) + 0.15 


    def __call__(self, frame, position):
        #Q： 试一下这个torch.no_grad是不是有点多余
        # with torch.no_grad():
        tensor = self.np2tensor(frame)
        s = ttime()

        res = self.model(tensor, position)

        spent = ttime() - s
        self.counter += 1
        # print(spent)
        self.inner_times += spent


        result = self.tensor2np(res)
        return result