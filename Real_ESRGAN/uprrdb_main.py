from torch2trt import TRTModule
from torch import nn as nn
import torch
from torch.nn import functional as F
from time import time as ttime
import numpy as np
import os, sys

# import files from local folder
root_path = os.path.abspath('.')
sys.path.append(root_path)
from process.utils import np2tensor, tensor2np
from config import configuration
from Real_ESRGAN.rrdb import RRDBNet


class UpRRDB2x(nn.Module):
    def __init__(self, rrdb_net_weight_path, adjust):
        super(UpRRDB2x, self).__init__()
        load_start = ttime()


        torch.cuda.empty_cache()
        if configuration.use_tensorrt:
            self.rrdb_model = TRTModule()
            self.rrdb_model.load_state_dict(torch.load(rrdb_net_weight_path))
            # don't use .eval().cuda() because it will raise a bug
            for param in self.rrdb_model.parameters():
                param.grad = None
        else:
            # Use original pretrained model
            self.rrdb_model = RRDBNet()
            model_weight = torch.load(os.path.join(configuration.weights_dir, configuration.model_name, configuration.architecture_name+'_weight.pth'))
            if "pro" in model_weight:
                del model_weight["pro"]
            self.rrdb_model.load_state_dict(model_weight, strict=True)
            self.rrdb_model.eval().cuda()


        self.adjust_double = 2*adjust
        print("torch2trt full load+prepare time %.3f s"%(ttime() - load_start))



    def forward(self, x, position):
        # 这个pad不知道现在有没有用
        # x = F.pad(x, (18, 18, 18, 18), 'reflect')           # pad最后一个倒数第二个dim各上下18个（总计36个）

        ######################## Neural Network Process #############################
        output = self.rrdb_model(x)
        #############################################################################


        ######################## After Process ######################################
        # 根据各个frame的position（上面，中间，下面，还是全部）来进行拆分adjust
        if position == 0:
            x = output[:, :, :-self.adjust_double, :]
        elif position == 1:
            x = output[:, :, self.adjust_double:-self.adjust_double, :]
        elif position == 2:
            x = output[:, :, self.adjust_double:, :]
        elif position == 3:
            # Full Frame Model
            x = output
        else:
            print("Error Position Type!")


        return (x * 255).round().clamp_(0, 255).byte()

        
    
class RealESRGAN_Scalar(object):
    
    def __init__(self, rrdb_weight_path, adjust):
        self.model = UpRRDB2x(rrdb_weight_path, adjust)
        if configuration.use_tensorrt:
            self.model = self.model.half()          # Must use half in tensorrt for float16, else the output is a black screen


    def __del__(self):
        # if self.counter:
        #     print("Inner time is %.2f s on %d, which is %.5f s per frame"%(self.inner_times, self.counter, self.inner_times/self.counter))
        return


    def __call__(self, frame, position):
        
        tensor = np2tensor(frame, pro=False)
        if configuration.use_tensorrt:
            tensor = tensor.half()          # Must use half in tensorrt for float16, else the output is a black screen


        res = self.model(tensor, position)


        result = tensor2np(res)
        return result