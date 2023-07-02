import os

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
