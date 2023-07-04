import os, time, sys
from config import configuration


# import from local folder
root_path_ = os.path.abspath('.')
sys.path.append(root_path_)
from process.single_video import parallel_process


def check_existence(dir, create=False):
    if not os.path.exists(dir):
        print("This " + dir + " file folder doesn't exist!")
        if not create:
            os._exit(0)
        else:
            os.mkdir(dir)



def mass_process(input_folder_dir, output_dir_parent):
    
    check_existence(input_folder_dir)
    check_existence(output_dir_parent, create=True)


    print("All files begin")
    for _, file in enumerate(sorted(os.listdir(input_folder_dir))):
        lists = file.split('.')
        target_name = ''.join(lists[:-1])


        # Find the name of input and ouput
        input_name = os.path.join(input_folder_dir, file)
        output_name = os.path.join(output_dir_parent, target_name + "_processed.mp4")
        print("We are super resolving {} and we will save it at {}".format(input_name, output_name))


        # Process the video
        # TODO: 利用log的report看看要不要减少partition的thread数量，毕竟相同视频类型都是相似的
        parallel_process(input_name, output_name, parallel_num=configuration.process_num)


        # After Processing
        print("After finish one thing, sleep for a moment!")
        time.sleep(5)
    os._exit(0)
        

