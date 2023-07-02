import os, time, sys
from config import configuration


# import from local folder
root_path_ = os.path.abspath('.')
sys.path.append(root_path_)
from process.single_process import parallel_process


def check_existence(dir, create=False):
    if not os.path.exists(dir):
        print("This " + dir + " file folder doesn't exist!")
        if not create:
            os._exit(0)
        else:
            os.mkdir(dir)


# def parse_args():
#     parser = argparse.ArgumentParser()
#     # parser.add_argument('--extract_subtitle', action='store_true')
#     global args
#     args = parser.parse_args()


def mass_process(input_folder_dir, output_dir_parent):
    # parse_args()
    
    check_existence(input_folder_dir)
    check_existence(output_dir_parent, create=True)


    print("All files begin")
    for _, file in enumerate(sorted(os.listdir(input_folder_dir))):
        lists = file.split('.')
        target_name = ''.join(lists[:-1])


        # Find name of input and ouput
        input_name = os.path.join(input_folder_dir, file)
        output_name = os.path.join(output_dir_parent, target_name + "_processed.mp4")
        print(input_name, output_name)


        # Process the video
        start = time.time()


        parallel_process(input_name, output_name, parallel_num=configuration.process_num)


        full_time_spent = int(time.time() - start)
        print("Total time spent for this video is %d min %d s" %(full_time_spent//60, full_time_spent%60))

        # TODO: 利用log的report看看要不要减少partition的thread数量，毕竟相同视频类型都是相似的

        print("After finish one thing, sleep for a moment!")
        time.sleep(5)
    os._exit(0)
        

