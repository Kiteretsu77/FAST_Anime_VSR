import torch.nn.functional as F
import torch.nn as nn
from torch2trt.torch2trt import *                                 
from torch2trt.module_test import add_module_test
import collections                     


# Debug collections.Sequence when I am generating the model for Real-ESRGAN
@tensorrt_converter('torch.nn.functional.interpolate', enabled=trt_version() >= '7.1')
@tensorrt_converter('torch.nn.functional.upsample', enabled=trt_version() >= '7.1')
def convert_interpolate_trt7(ctx):                                     
    #parse args                     
    input = get_arg(ctx, 'input', pos=0, default=None) 
    size = get_arg(ctx, 'size', pos=1, default=None)
    scale_factor=get_arg(ctx, 'scale_factor', pos=2, default=None)
    mode = get_arg(ctx, 'mode', pos=3, default='nearest')
    align_corners = get_arg(ctx, 'align_corners', pos=4, default=None)

    input_dim = input.dim() - 2
    
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output = ctx.method_return
    layer = ctx.network.add_resize(input=input_trt)

    shape = size
    if shape != None:
        if isinstance(shape, collections.abc.Sequence):
            shape = [input.size(0), input.size(1)] + list(shape)
            shape = make_size_wrapper(shape)
        else:
            shape = [input.size(0), input.size(1)] + [shape] * input_dim
            shape = make_size_wrapper(shape)

        # layer.shape = shape (old, static shape)
        layer.set_input(1, shape._trt)

    scales = scale_factor
    if scales != None:
        if not isinstance(scales, collections.abc.Sequence):
            scales = [scales] * input_dim
        layer.scales = [1, 1] + list(scales)

    resize_mode = mode
    if resize_mode.lower() in ["linear","bilinear","trilinear"]:
        layer.resize_mode = trt.ResizeMode.LINEAR
    else:
        layer.resize_mode=trt.ResizeMode.NEAREST

    if align_corners != None:
        if trt_version() > '8.0':
            if align_corners:
                layer.coordinate_transformation = trt.ResizeCoordinateTransformation.ALIGN_CORNERS
        else:
            layer.align_corners = align_corners

    output._trt = layer.get_output(0)


