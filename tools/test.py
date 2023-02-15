import sys
from moviepy.editor import *


def sec2foramt(time):

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



input_file = r"C:\Users\hikar\Desktop\anime_src\video2.mp4"
clip = VideoFileClip(input_file)
print(clip.duration)

divide_time = int(clip.duration // 2)
middle_time = sec2foramt(divide_time)

ffmpeg_cmd1 = "ffmpeg -ss 00:00:00 -to " + middle_time + " -accurate_seek -i " + input_file +  " -codec copy -avoid_negative_ts 1 tmp/part1.mp4"
os.system(ffmpeg_cmd1)

ffmpeg_cmd2 = "ffmpeg -ss " + middle_time + " -accurate_seek -i " + input_file +  " -codec copy -avoid_negative_ts 1 tmp/part2.mp4"
os.system(ffmpeg_cmd2)




output_name1 = "part1.mp4"
output_name2 = "part2.mp4"
if os.path.isfile("tmp/target.txt"):
    os.remove("tmp/target.txt")
if os.path.isfile("tmp/output.mp4"):
    os.remove("tmp/output.mp4")

file = open("tmp/target.txt", "a")
file.write("file part1.mp4\n")
file.write("file part2.mp4")
file.close()
target_output = "tmp/output.mp4"

ffmpeg_combine_cmd = "ffmpeg -f concat -i tmp/target.txt -c copy " + target_output
os.system(ffmpeg_combine_cmd)



