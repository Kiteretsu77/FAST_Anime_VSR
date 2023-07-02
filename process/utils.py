import os

def check_input_support(file):
    lists = file.split('.')
    format = lists[-1]

    accepted_format = ["mp4", "mkv", "mov"]
    if format not in accepted_format:
        print("This format is not supported!")
        os._exit(0)

    return format
