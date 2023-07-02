import collections, math
import threading, sys, torch, cv2
from random import uniform
from multiprocessing import Queue
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from moviepy.editor import VideoFileClip
from time import time as ttime, sleep
import os, time, shutil, random
import numpy as np
from multiprocessing import Process

# import file from other directory
from Real_CuGAN.upcunet_main import RealCuGAN_Scalar

from threading import Lock
s_print_lock = Lock()

# import from local folder
root_path_ = os.path.abspath('.')
sys.path.append(root_path_)


class UpScalerMT(threading.Thread):
    def __init__(self, id, inp_q, res_q, device, model, p_sleep, nt):
        '''
            Multi-Thread Processing
        '''
        # threading.Thread.__init__(self)
        super(UpScalerMT, self).__init__()
        self.id = id
        self.device = device
        self.inp_q = inp_q
        self.res_q = res_q
        self.model = model
        self.nt = nt
        self.p_sleep = p_sleep

        #下面几个是用于时间统计的
        self.cost_list = []
        self.total_cost_time = 0
        self.total_counter_ = 0


    def __del__(self):
        '''
            Report time 
        '''
        if self.total_counter_ != 0:
            with s_print_lock:
                print("UpScalerMT #" + str(self.id) + " report:")
                # print("\tFull Exe Cost: " + str(round(sum(self.cost_list), 2)) + "s on " + str(len(self.cost_list)) + " frames and in average " + str(round(sum(self.cost_list) / len(self.cost_list), 4 )) + "s")
                print("\tFull Exe Cost: " + str(round(self.total_cost_time, 2)) + "s on " + str(
                    self.total_counter_) + " frames and in average " + str(
                    round(self.total_cost_time / self.total_counter_, 4)) + "s \n")

    def execute(self, np_frame, position):
        '''
            Send data(frames) to the model
        '''
        return self.model(np_frame, position)


    def inference(self, tmp):
        idx, position, np_frame = tmp

        
        ####################### RealCuGAN Execuation #####################
        full_exe_start_time = ttime()

        try:
            res = self.execute(np_frame, position)

        except Exception:
            print("Exception! Execute Exception problem, try to execute again")
            res = self.execute(np_frame, position)

        full_exe_end_time = ttime()
        full_exe_time_spent = full_exe_end_time - full_exe_start_time
        ################################################################

        self.total_cost_time += full_exe_time_spent
        self.total_counter_ += 1
        if self.nt > 1:
            # sleep adjustment, Must have here else there may be bug
            sleep(uniform(self.p_sleep[0], self.p_sleep[1]))

        return (idx, position, res)


    def run(self):
        while True:
            tmp = self.inp_q.get()
            # print(self.device + " inp_q size is ", self.inp_q.qsize())
            if tmp == None:
                break

            if self.res_q.full():
                print("Exception!!! res_q is full, Needs more optimization! Will pause some time!")
                # sleep(1)
            self.res_q.put(self.inference(tmp))



class VideoUpScaler(object):
    def __init__(self, configuration):

        ###################################### Important Params ##################################################################################
        self.writer = None    # Used in Call function
        self.nt = configuration.nt
        self.scale = configuration.scale
        self.n_gpu = configuration.n_gpu  # number of threads that each GPU use
        self.encode_params = configuration.encode_params
        self.decode_sleep = configuration.decode_sleep
        self.now_idx = 0
        self.loop_counter = 0
        self.total_frame_number = 0
        self.decode_fps = configuration.decode_fps
        self.height = None
        self.device = configuration.device

        if configuration.full_model_num == 0:
            self.max_cache_loop = max(int((configuration.nt + configuration.full_model_num) * configuration.n_gpu * configuration.Queue_hyper_param),
                                      int(2 * configuration.n_gpu * configuration.Queue_hyper_param))
        else:
            self.max_cache_loop = max(int((configuration.nt//3 + configuration.full_model_num) * configuration.n_gpu * configuration.Queue_hyper_param),
                                            int(2 * configuration.n_gpu * configuration.Queue_hyper_param))
        print("max_cache_loop size is ", self.max_cache_loop)
        #################################################################################################################################

        ################################### load model #################################################################
        self.check_weight_support()

        unet_full_path_partition = "weights/unet_full_weight_trt_" + configuration.unet_partition_name + "_float16.pth"

        if configuration.full_model_num != 0:
            unet_full_path_full_frame = "weights/unet_full_weight_trt_" + configuration.unet_full_name + "_float16.pth"

        #################################################################################################################

        ######################## similar frame optimization #######################
        self.skip_counter_ = 0
        self.reference_frame = [None, None, None]
        self.reference_idx = [-1, -1, -1]
        self.parition_processed_num = 0
        self.MSE_range = configuration.MSE_range
        self.Max_Same_Frame = configuration.Max_Same_Frame
        self.mse_learning_rate = configuration.mse_learning_rate
        ############################################################################

        ########################## Image Crop & Momentum ##########################################################################
        self.adjust = configuration.adjust
        self.left_mid_right_diff = configuration.left_mid_right_diff # 这个理解起来就是第一个最右/下侧多2， 中间两边都少2（同少4），最后一个左/上边多2

        self.full_frame_cal_num = 0
        self.full_model_num = configuration.full_model_num  # number of full model available

        self.momentum = configuration.momentum
        self.momentum_reference_size = 3 # queue size
        self.times2switchFULL = 0
        self.momentum_used_num = 0
        self.momentum_reference = collections.deque([False]*self.momentum_reference_size, maxlen=self.momentum_reference_size)
        ############################################################################################################################

        ############################### MultiThread And MultiProcess ###################################
        # Full Frame
        self.inp_q_full = None
        if configuration.full_model_num != 0:
            Full_Frame_Queue_size = int( (self.full_model_num + 1) * self.n_gpu * configuration.Queue_hyper_param)
            self.inp_q_full = Queue(Full_Frame_Queue_size) # queue of full frame
            print("Total FUll Queue size is ", Full_Frame_Queue_size)

        # Sub Frame 
        self.inp_q = None
        if self.nt != 0:
            Divided_Block_Queue_size = int( (self.nt) * self.n_gpu * configuration.Queue_hyper_param//2)
            print("Total Divided_Block_Queue_size is ", Divided_Block_Queue_size)
            self.inp_q = Queue(Divided_Block_Queue_size)  # 抽帧缓存上限帧数
        
        # In total
        Res_q_size = int( (self.nt + self.full_model_num) * self.n_gpu * configuration.Queue_hyper_param//2)
        print("res_q size is ", Res_q_size)
        self.res_q = Queue(Res_q_size)  # Super-Resolved Frames Cache
        self.idx2res = collections.defaultdict(dict)
        ###################################################################################################


        ############################# Model Preparation ###############################################################################

        # 目前先搞少量的full model place
        for _ in range(self.full_model_num):
            model = RealCuGAN_Scalar(unet_full_path_full_frame, self.device, self.adjust)
            upscaler_full = UpScalerMT("FULL", self.inp_q_full, self.res_q, "full", model, configuration.p_sleep, 1)
            upscaler_full.start()


        for id in range(self.nt):
            model = RealCuGAN_Scalar(unet_full_path_partition, self.device, self.adjust)
            upscaler = UpScalerMT(id, self.inp_q, self.res_q, self.device, model, configuration.p_sleep, self.nt)
            upscaler.start()
        ###############################################################################################################################

    def check_weight_support(self):
        pass

    def frame_write(self):
        # 从res_q中提取内容， 并整合成一张大图内容

        #Step1： 写入暂存器，因为多进程多线程的结果是不均匀出来的
        while True:  # 取出处理好的所有结果
            if self.res_q.empty():
                break
            iidx, position, res = self.res_q.get()
            self.idx2res[iidx][position] = res


        #Step2: 把暂存器的内容写到writer中
        while True:  # 按照idx排序写帧
            if not self.res_q.empty():
                iidx, position, res = self.res_q.get()
                self.idx2res[iidx][position] = res

            #这里一定保证是sequential的，所以repeat frame前面的reference完全有加载
            if self.loop_counter == self.max_cache_loop:  #####这个系数也要config管理！#####
                self.writer.close()
                print("Ends at frame ", self.now_idx)
                print("P: Continuously not found, end the program and store what's stored")
                os._exit(0)

            # if self.now_idx >= 2063:
            #     print(self.now_idx, self.idx2res[self.now_idx])

            if not all(i in self.idx2res[self.now_idx] for i in [0,1,2]) and not 3 in self.idx2res[self.now_idx]:
                self.loop_counter += 1
                break
            self.loop_counter = 0


            #############################################下面确保了frame的所有部分都是在的################################################
            if self.now_idx % 50 == 0:
                print("Had Written frames:%s" %self.now_idx)

            # 3种类型的crop处理
            if 3 not in self.idx2res[self.now_idx]:
                crops = []
                for i in [0, 1, 2]:
                    if isinstance(self.idx2res[self.now_idx][i], int):
                        target_idx = self.idx2res[self.now_idx][i]
                        if 3 in self.idx2res[target_idx]:
                            #这里就说明目标是一整块没有被切分成block，需要提取一下
                            crops.append(self.full_crop(self.idx2res[target_idx][3], i))
                        else:
                            crops.append(self.idx2res[target_idx][i])
                    else:
                        crops.append(self.idx2res[self.now_idx][i])
                combined_frame = self.combine(*crops)  # adjust是完全固定的值
            else:
                #TODO: 如果后面FULL也要算MSE的话，这里就要写的更加复杂了
                combined_frame = self.idx2res[self.now_idx][3]


            # TODO: 如果是用之前的ref的时候，直接提取之前的计算结果
            self.writer.write_frame(combined_frame)


            ###为了程序简单，多了Max_Same_Frame帧的cache
            if self.now_idx > self.Max_Same_Frame:
                del self.idx2res[self.now_idx - self.Max_Same_Frame]

            self.now_idx += 1


    def combine(self, crop1, crop2, crop3):
        #这里combine images together！！！！
        # cv2.imwrite("res/crop" + str(self.now_idx) +"_1.png", crop1[:-6, :, :])
        # cv2.imwrite("res/crop" + str(self.now_idx) +"_2.png", crop2[6:-6, :, :])
        # cv2.imwrite("res/crop" + str(self.now_idx) +"_3.png", crop3[6:, :, :])
        # print(crop1.shape, crop2.shape, crop3.shape)
        # temp = np.concatenate((crop1[:-12, :, :], crop2[12:-12, :, :], crop3[12:, :, :]))
        # cv2.imwrite("res/img" + str(self.now_idx) + ".png", temp)
        # return np.concatenate((crop1[:-12, :, :], crop2[12:-12, :, :], crop3[12:, :, :]))
        return np.concatenate((crop1, crop2, crop3))


    def full_crop(self, frame, position):
        # 主要应用在mse history搜寻的时候，如果是一整块，就只能分割一下
        crop_base = self.height // 3
        # TODO: 这里要optimize一下，找个地方存一下计算结果
        if position == 0:
            return frame[:2*(crop_base+self.left_mid_right_diff[0]), :, :] # :324
        elif position == 1:
            return frame[2*(crop_base-self.left_mid_right_diff[1]) : 4*crop_base+2*self.left_mid_right_diff[1], :, :] # 324:636
        elif position == 2:
            return frame[4*crop_base+2*self.left_mid_right_diff[1]:, :, :] #636:


    def crop(self, img):
        # TODO: 这里要optimize一下，找个地方存一下计算结果
        crop_base = self.height // 3  # 160 for 480p
        crop1 = img[:crop_base + 2*self.adjust + self.left_mid_right_diff[0], :, :] # :168
        crop2 = img[crop_base-self.left_mid_right_diff[1]-2*self.adjust : 2*crop_base+self.left_mid_right_diff[1]+2*self.adjust, :, :] # 156:324
        crop3 = img[-(crop_base + 2*self.adjust + self.left_mid_right_diff[2]):, :, :] # -168:
        # print(crop1.shape, crop2.shape, crop3.shape)
        # cv2.imwrite('split/' + str(time.time()) + '_1.png', cv2.cvtColor(crop1, cv2.COLOR_BGR2RGB))
        # cv2.imwrite('split/' + str(time.time()) + '_2.png', cv2.cvtColor(crop2, cv2.COLOR_BGR2RGB))
        # cv2.imwrite('split/' + str(time.time()) + '_3.png', cv2.cvtColor(crop3, cv2.COLOR_BGR2RGB))
        return (crop1, crop2, crop3)


    def queue_put(self, idx, i, crop ):
        if i == 0:
            crop = crop[:-self.adjust, :, :]
        elif i == 1:
            crop = crop[self.adjust:-self.adjust, :, :]
        elif i == 2:
            crop = crop[self.adjust:, :, :]

        # print(crop.shape)
        self.parition_processed_num += 1
        self.inp_q.put((idx, i, crop))



    def __call__(self, input_path, output_path):
        ############################### Build PATH && INIT ############################################
        video_format = input_path.split(".")[-1]
        os.makedirs(os.path.join(root_path_, "tmp"), exist_ok=True)
        tmp_path = os.path.join(root_path_, "tmp", "%s.%s" % (int(ttime()*1000000), video_format))
        os.link(input_path, tmp_path)
        objVideoreader = VideoFileClip(filename=tmp_path)
        self.width, self.height = objVideoreader.reader.size
        # scale = 1的adjust放在了writer init后面


        fps = objVideoreader.reader.fps
        if self.decode_fps == -1:
            # use original fps as decode fps
            self.decode_fps = fps
        self.total_frame_number = int(self.decode_fps * (objVideoreader.reader.nframes/fps))
        if_audio = objVideoreader.audio
        #############################################################################################

        

        ################################### Build Video Writer ########################################################################################
        if if_audio:
            tmp_audio_path = "%s.m4a" % tmp_path
            objVideoreader.audio.write_audiofile(tmp_audio_path, codec="aac")
            # 得到的writer先给予audio然后再一帧一帧的写frame
            self.writer = FFMPEG_VideoWriter(output_path, (self.width * self.scale, self.height * self.scale), self.decode_fps, ffmpeg_params=self.encode_params, audiofile=tmp_audio_path)
        else:
            self.writer = FFMPEG_VideoWriter(output_path, (self.width * self.scale, self.height * self.scale), self.decode_fps, ffmpeg_params=self.encode_params)
        # 如果是1x scale，这个后面开始就要shrink了
        if self.scale != 2:
            self.width = int(self.width * (self.scale/2))
            self.height = int(self.height * (self.scale/2))
        ##############################################################################################################################################


        mse_total_spent = 0
        video_decode_loop_start = ttime()
        ######################################### video decode loop #######################################################
        for idx, frame in enumerate(objVideoreader.iter_frames(fps=self.decode_fps)): # 删掉了target fps
            
            if self.scale != 2:
                # if scale = 1/1.5, adjust output size at the beginning
                frame = cv2.resize(frame, (self.width, self.height)) # , interpolation=cv2.INTER_LANCZOS4

            if idx % 50 == 0 or int(self.total_frame_number) == idx:
                # 以后print这边用config统一管理
                print("total frame:%s\t video decoded frames:%s"%(int(self.total_frame_number), idx))
                sleep(self.decode_sleep)  # 否则解帧会一直抢主进程的CPU到100%，不给其他线程CPU空间进行图像预处理和后处理
                    # 目前nt=1的情况来说，不写也无所谓

            # check if NN process too slow to catch up the frames decoded
            decode_processed_diff = idx - self.now_idx
            if decode_processed_diff >= 650:
                #TODO: 这个插值也要假如config中
                self.frame_write()
                if decode_processed_diff >= 1000:
                    # Have to do this else it's possible to raise bugs
                    print("decode too slow, needs to sleep 0.4s")
                    sleep(0.4)


            ######################### use MSE to judge whether it's highly overlapped #########################

            if self.MSE_range != -1:
                if self.nt > 0 and self.times2switchFULL <= 0:
                    ############################################## Split Frame  ########################################
                    (crop0, crop1, crop2) = self.crop(frame)  # adjust要固定死, cropX后面是用eval控制的
                    ####################################################################################################

                    changed_status = [0, 0, 0]
                    for i in range(3):
                        crop = eval("crop%s"%i)
                        if self.reference_frame[i] is None:
                            self.reference_frame[i] = crop[:, :, 0]
                            self.reference_idx[i] = idx
                            self.queue_put(idx, i, eval("crop%s" % i)) # 这里put了以后，最下面就没必要再put了

                        else:
                            # 这边我测过，用absolute error速度也不会快
                            frame_err = np.square(self.reference_frame[i] - crop[:, :, 0])
                            frame_err = np.mean(frame_err, dtype=np.float32)  # 这里不能用float16，会更加慢（可能是有一个cast的过程）

                            if not (idx - self.reference_idx[i] < self.Max_Same_Frame):
                                # overflow, should decrease
                                self.reference_frame[i] = eval("crop%s" % i)[:, :, 0]
                                self.reference_idx[i] = idx
                                changed_status[i] = 1000 # 设置成1000，类似于调整成无限大，然后为了momentum

                            else:
                                if frame_err <= self.MSE_range:
                                    # 发现具有高相似度，所以直接put到idx2res中的暂存器就好
                                    self.idx2res[idx][i] = self.reference_idx[i]
                                    self.skip_counter_ += 1

                                else:
                                    # 因为现在两帧之间的差距大于设定mse值了，所以还是走到下一帧好
                                    self.reference_frame[i] = eval("crop%s" % i)[:, :, 0]
                                    self.reference_idx[i] = idx

                                    # 目前来说还是都变化比较大才有加载的意义
                                    changed_status[i] = frame_err

                    if self.full_model_num != 0 and all(status > self.MSE_range for status in changed_status):
                        # 如果现在有full mode，然后每个error都大于MSE，这个地方，丢一整个frame进去
                        # TODO: 这里还是可以加进去动态MSE的算式&&full比例控制算法，portion就这个单独算就行
                        self.full_frame_cal_num += 1

                        # if self.inp_q_full.full():
                        #     print("Exception!!! inp_q_full is full")
                        self.inp_q_full.put((idx, 3, frame))

                        # 这样子后面回来相当于又重新开始计算，momentum会比写的数字效果+1
                        # TODO: 测试一下momentum_reference_size和self.momentum用什么值好
                        if all(status > 5 for status in changed_status):
                            self.momentum_reference.append(True)
                            if sum(self.momentum_reference) == self.momentum_reference_size: # All True
                                self.times2switchFULL = self.momentum
                                self.reference_frame = [None, None, None]
                                self.reference_idx = [-1, -1, -1]
                                self.momentum_reference.extend([False] * self.momentum_reference_size)
                        else:
                            self.momentum_reference.append(False)

                        continue

                    # 如果不走full情况，根据frame_err来塞
                    self.momentum_reference.append(False)
                    # if self.inp_q.full(): # 这个full的参考价值不大，但是未来也是可以继续注意的
                    #     print("Exception!!! inp_q is full")
                    for i in range(3):
                        # 所有error都放进去
                        if changed_status[i] > self.MSE_range:
                            self.queue_put(idx, i, eval("crop%s"%i))
                else:
                    # TODO： momentum也还是可以加上去，不过这样子mse动态调整的数字就要变小
                    # print("need to use momentum")

                    self.full_frame_cal_num += 1
                    self.momentum_used_num += 1
                    #TODO: 我这里是加上inference还是就直接momentum懒得计算，直接丢算了 ！！！TODO x2
                    # 方案1: 暂时就这样子过了就好，顺便把全部之前的reference_frame设置为None
                    # 方案2： 写的麻烦一点，再搞个reference 和 idx
                    if self.nt > 0:
                        self.times2switchFULL -= 1
                    self.inp_q_full.put((idx, 3, frame))

            ####################################################################################################

            #TODO: 这里每3帧才写入一次
            # sleep(self.decode_sleep) # 这个要放在frame_write前面
            if idx % 4 == 0:
                #only write in even number, tries to load 3 frames each time before taking this
                self.frame_write()

        print("All Frames are put into section")
        #########################################################################################################


        ################################################ 后面残留的计算 ##################################################
        idx += 1 # 调整成frames总数量
        #等待所有的处理完,最后读取一遍全部的图片
        while True:
            self.frame_write()

            # if self.inp_q is not None and self.inp_q.qsize() == 0 and self.res_q.qsize() == 0 and idx == self.now_idx:
            #     break
            # elif self.res_q.qsize() == 0 and idx == self.now_idx:
            #     break

            if idx == self.now_idx:
                if self.inp_q is not None:
                    assert(self.inp_q.qsize() == 0)
                assert(self.res_q.qsize() == 0)
                break

            # TODO: 这个目前发现不运行就会出问题, 要不要用decode sleep统一处理, 这个bug是不是res_q满载了的原因
            sleep(self.decode_sleep) # 原本0.01

        print("Final image index is ", self.now_idx)

        for _ in range(self.nt * self.n_gpu):  # 全部结果拿到后，关掉模型线程
            self.inp_q.put(None)
        for _ in range(self.full_model_num  * self.n_gpu):
            self.inp_q_full.put(None)

        # close writer to save all stuff
        self.writer.close()
        if if_audio:
            os.remove(tmp_audio_path)
        
        # os.remove(tmp_path)  # 把视频都暂时link到tmp文件中，最后还是要删的

        video_decode_loop_end = ttime()
        ################################################################################################################

        ##################################### 分析汇总 ##################################################################

        full_time_spent = video_decode_loop_end - video_decode_loop_start
        total_exe_fps = self.total_frame_number / full_time_spent
        full_frame_portion = self.full_frame_cal_num / self.total_frame_number
        partition_saved_portion = self.skip_counter_/(self.total_frame_number*3)
        
        print("Input path is %s and the report is the following:"%input_path)
        if full_time_spent < 60:
            print("Done! Total time cost:", full_time_spent)
        else:
            print("Done! Total time cost: %d min %d s" %(full_time_spent//60, full_time_spent%60))
        print("Counting Saved frame, the fps is %.2f which is %.2f scale on fps" %(total_exe_fps, total_exe_fps/self.decode_fps))
        print("The Number of partitions put into small Upscaler is %d which is %.2f %%" % (
                self.parition_processed_num, 100 * self.parition_processed_num / (self.total_frame_number * 3)))
        print("Saved frames number: %d partitions which is %.2f %%" %(self.skip_counter_, 100*partition_saved_portion))
        # print("mse_total_spent %.3f s"%(mse_total_spent))
        print("Total full_frame_cal_num is %.2f which is %.2f %%" %(self.full_frame_cal_num, 100*full_frame_portion))
        print("Total momentum used num is ", self.momentum_used_num)
        ################################################################################################################
        

        ##################################### Generate Final Report ####################################################

        report = {}
        report["input_path"] = input_path
        report["full_time_spent"] = full_time_spent
        report["total_exe_fps"] = total_exe_fps
        report["performance_scale"] = total_exe_fps/self.decode_fps
        report["parition_processed_num"] = self.parition_processed_num
        report["skip_counter"] = self.skip_counter_
        report["full_frame_cal_num"] = self.full_frame_cal_num
        report["momentum_used_num"] = self.momentum_used_num

        return report



