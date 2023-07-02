# 头三个一定要按照这个path
import tensorrt
from torch2trt import torch2trt
import torch 
# 上面三个不按照这个顺序就会有bug

from torch import nn as nn
from torch.nn import functional as F
from time import time as ttime
from torch2trt import TRTModule
import cv2
import numpy as np
import time
import os, sys
from time import time as ttime, sleep
import argparse
import shutil
import requests


# Import from local folder
root_path = os.path.abspath('.')
sys.path.append(root_path)
from config import configuration


from Real_CuGAN.cunet import UNet1, UNet2, UNet_Full



######################################### helper function  #################################################################
def tensor2np(tensor):
    # 这边看看还有没有什么能够提升的点，耗时实在太长了
    return (np.transpose(tensor.squeeze().cpu().numpy(), (1, 2, 0)))

def np2tensor(np_frame):
    # return torch.from_numpy(np.transpose(np_frame, (2, 0, 1))).unsqueeze(0).to("cuda:0").float() / 255
    ###### pro mode
    return torch.from_numpy(np.transpose(np_frame, (2, 0, 1))).unsqueeze(0).to("cuda:0").float() / (255 / 0.7) + 0.15
#############################################################################################################################


# This will be set in config in the future, maybe
full_sample_img_dir = "weight_generation/sample_input.png"



class Generator:
    def __init__(self, sample_input_dir):
        if not os.path.exists(sample_input_dir):
            print("no such sample_input_dir exists: ", sample_input_dir)
            os._exit(0)

        img = cv2.imread(sample_input_dir)
        (self.h, self.w, _) = img.shape
        print("height & width is ", self.h, self.w)
        self.sample_input = np.array(img)

        self.dont_calculate_transform = None

    def pre_process(self, tensor):
        # 这个np2tensor别忘记了
        if not self.dont_calculate_transform:
            tensor = np2tensor(tensor)
        
        #暂时默认全部都是偶数的情况，这点要特别注意
        intput = F.pad(tensor, (18, 18, 18, 18), 'reflect')  # pad最后一个倒数第二个dim各上下18个（总计36个）
        input = intput.cuda()
        # print("After pre-process, the shape is ", input.shape)
        return input


    def after_process(self, x):

        ####Q: 是不是add以后这边变成cpu更加节约时间

        # h0 = 480
        # w0 = 640
        # ph = ((h0 - 1) // 2 + 1) * 2 # 应该是用来确认奇数偶数的
        # pw = ((w0 - 1) // 2 + 1) * 2
        # if w0 != pw or h0 != ph:
        #     x = x[:, :, :h0 * 2, :w0 * 2] #调整成偶数的size

        if self.h%2 != 0 or self.w%2 != 0:
            print("ensure that width and height to be even number")
            os._exit(0)

        ########目前默认是pro mode
        temp =  ((x - 0.15) * (255/0.7)).round().clamp_(0, 255).byte()
        # print("After after-process, the shape is ", temp.shape)
        return temp


    def testify(self, img, test_number, partition_status):
        if not os.path.exists(args.args.test_dir):
            print("No such test exists")
            return

        if os.path.exists("Sample_output"):
            shutil.rmtree("Sample_output")
        os.mkdir("Sample_output")

        self.dont_calculate_transform = True

        print("----------------------Test Page-----------------------")
        load_start = ttime()

        torch.cuda.empty_cache()
        unet_model_full = TRTModule()
        unet_model_full.load_state_dict(torch.load(self.weight_store_dir + 'unet_full_weight_trt_' + self.base_name + '_float16.pth'))
        # unet_model_full = unet_model_full.cuda().eval()
        for param in unet_model_full.parameters():
            param.grad = None

        print("torch2trt load time ", ttime() - load_start)


        # Running {num} number of images to test the performance
        total_time = 0
        inner_time = 0

        for idx, filename in enumerate(os.listdir(args.test_dir)):
            if idx == test_number:
                break
            file_dir = os.path.join(args.test_dir, filename)
            if not os.path.exists(file_dir):
                print("No such file exists: ", file_dir)
                os._exit(0)
            img = cv2.imread(file_dir)
            h, w, _ = img.shape

            if self.h > h or self.w > w:
                print("Larger size than test image")
                os._exit(0)
            
            target_height = self.h * 3 if partition_status else self.h  # 只有是partition的时候才adjust height

            img = cv2.resize(img, (self.w, target_height), interpolation = cv2.INTER_AREA)
            img = img[:self.h, :self.w, :] # 需要多少裁剪多少

            exe_start = ttime()
            if self.dont_calculate_transform: # dont_calculate_transform是True的时候np2tensor就在pre_process以外调用就行
                tensor = np2tensor(img).half()
            
            
            ######################### core image model generation #########################
            a = ttime()

            tensor_input = self.pre_process(tensor)
            unet_full_output = unet_model_full(tensor_input)
            final_output = self.after_process(unet_full_output)

            inner_time += ttime() - a
            ################################################################################

            # tensor2np edit
            final_output = tensor2np(final_output)

            print(filename, final_output.shape)
            total_time += (ttime() - exe_start) 

            if args.store_all_testify:
                # store all results
                cv2.imwrite("Sample_output/" + str(idx) + ".png", final_output)

            elif idx == test_number//2:
                print("Try to save a result at idx ", idx)
                cv2.imwrite("Sample_output/testify_sample_output.png", final_output)


        print("Outer: " + str(total_time) + " on " + str(test_number) + " frames, so in average, the speed is %.5f s"%(total_time/test_number))
        print("Inner: " + str(inner_time) + " on " + str(test_number) + " frames, so in average, the speed is %.5f s"%(inner_time/test_number))

        cv2.imwrite("Sample_output/final_sample_output.png", final_output)
        print("Testify image saved as sample_output.png")


    def unet_full_weight_transform(self, base_name, input):
        #input先假设就是正常的size 480/720 etc.
        if input is None:
            os._exit(0)

        input = self.pre_process(input)

        torch.cuda.empty_cache()
        unet_full = UNet_Full()
        if args.my_model:
            print("Use my own model!")
            checkpoint_g = torch.load(self.org_weight_store_dir + "my_model.pth", map_location="cpu")
            unet_full.load_state_dict(checkpoint_g['model_state_dict'], strict=True)

        else:
            unet_full_weight = torch.load(self.org_weight_store_dir, map_location="cpu") #pro-denoise3x-up2x.pth
            del unet_full_weight["pro"]
            unet_full.load_state_dict(unet_full_weight, strict=True)

        unet_full.eval().cuda()

        
        with torch.no_grad():
            print('converting unet full trt...')
            if args.int8_mode:
                from calibration import ImageFolderCalibDataset
                print("Use int8 mode")
                mode = "int8"
                print("intput shape is ", input.shape)
                dataset = ImageFolderCalibDataset("imgs/", self.h, self.w)
                unet_full_trt_model = torch2trt(unet_full, [input], int8_mode=True, int8_calib_dataset=dataset)
            else:
                print("Use float16 mode in TensorRT")
                mode = "float16"
                input = input.half() 
                unet_full = unet_full.half()
                print("Generating the TensorRT weight form ........")
                unet_full_trt_model = torch2trt(unet_full, [input], fp16_mode=True)

        print("Finish generating the tensorRT weight")
        torch.save(unet_full_trt_model.state_dict(), self.weight_store_dir + 'unet_full_weight_trt_' + base_name + '_'+mode+'.pth')


        # 测试一下output
        unet_full_output = unet_full_trt_model(input)
        print("tested output shape is ", unet_full_output.shape)

        print("unet full weigtht transforms Finished!")
        return unet_full_output
        

    def weight_generate(self):
        # 如果要从头开始weight生成的话，dont_calculate_transform为false；只是image大量测试，就用true就行
        self.dont_calculate_transform = False

        self.unet_full_weight_transform(self.base_name, self.sample_input)

        print("weight generate succeed!")


    def run(self, partition = False):
        
        # some global setting
        self.base_name = str(int(self.w)) + "X" + str(int(self.h))
        self.org_weight_store_dir = "weights/cunet_weight.pth"
        self.weight_store_dir = "weights/"

        ###################################################################################

        # 生成新的weight
        if not args.only_testify:
            self.weight_generate()

        #试一下batch
        if args.test_dir != "":
            self.testify(self.sample_input, test_number=100, partition_status=partition)


def generate_partition_frame(full_frame_dir):
    # Cut the frame to three fold for Video redundancy acceleartion in inference.py

    img = cv2.imread(full_frame_dir)
    h, w, _ = img.shape
    partition_height = (h//3) + 8 # TODO: 这个+8只是一个简单的写法，实际上应该更加dynamic的分配

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    partition_img = img[:partition_height, :,:] # 第一个是width，第二个是height
    print("Size after crop is ", partition_img.shape)
    partition_frame_dir = "weight_generation/1in3.png"
    cv2.imwrite(partition_frame_dir, partition_img)

    return partition_frame_dir


def crop_image(img_dir, target_h, target_w):
    img = cv2.imread(img_dir)
    print("Input image size is ", img.shape)

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Check if image is over size
    h, w, _ = img.shape
    if h < target_h or w < target_w:
        print("Such height and/or width is not supported, please use a larger sample input")
        os._exit(0)

    croped_img = img[:target_h, :target_w,:] # 第一个是height，第二个是width
    print("Size after crop is ", croped_img.shape)
    cv2.imwrite(full_sample_img_dir, croped_img)


def tensorrt_transform_execute(target_h, target_w, img_dir="weight_generation/full_sample.png"):
    start = time.time()

    # Crop image to desired height and width
    crop_image(img_dir, target_h, target_w)

    # Generate full frame
    if not args.only_partition_frame:
        ins = Generator(full_sample_img_dir) 
        ins.run()
        print("Full Frame weight generation Done!")


    # Generate partition frame
    if not args.only_full_frame:
        partition_img_dir = generate_partition_frame(full_sample_img_dir)
        ins = Generator(partition_img_dir) 
        ins.run(partition = True)
        print("Partition Frame generation Done!")


    print("Total time spent is %d s" %(int(time.time() - start)))


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--only_full_frame", action='store_true', help="")
    parser.add_argument("--only_partition_frame", action='store_true', help="")
    parser.add_argument('--test_dir', type=str, default="", help=" If you want to have personal test input to calibrate, you can use this one.")


    parser.add_argument("--my_model", action='store_true', help="to load personal trainned model")
    parser.add_argument("--only_testify", action='store_true', help="")
    parser.add_argument("--store_all_testify", action='store_true', help="")
    
    # parser.add_argument("--int8_mode", action='store_true')   // Very unstable, so we don't recommend using it.
    

    global args
    args = parser.parse_args()

    args.int8_mode = False

def check_file():
    if not os.path.exists("weights/cunet_weight.pth"):
        print("There isn't cunet_weight.pth under weights folder")
        

        # Automatically download Code, but if you want other weight, like less denoise, please go see https://drive.google.com/drive/folders/1jAJyBf2qKe2povySwsGXsVMnzVyQzqDD
        print("We will automatically download one from CuNet repository google drive!!!")
        url = "https://drive.google.com/u/0/uc?id=1hc1Xh_1qBkU4iGzWxkThpUa5_W9t7GZ_&export=download"
        r = requests.get(url, allow_redirects=True)
        open('weights/cunet_weight.pth', 'wb').write(r.content)
        # shutil.move('weights/cunet_weight.pth', 'weights/cunet_weight.pth')

        print("Finish downloading!")


def generate_weight(lr_h = 540, lr_width = 960):
    parse_args()
    check_file()

    tensorrt_transform_execute(lr_h, lr_width)
    

if __name__ == "__main__":
    generate_weight()


