import subprocess

def cut_8k_video_lossless(input_path, output_path, start_time, end_time):

    ffmpeg_path = r"C:\ffmpeg\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe"

    command = [
        ffmpeg_path,
        "-ss", start_time,
        "-to", end_time,
        "-i", input_path,
        "-map", "0:v:0",   # video only
        "-map", "0:a:0",   # audio only
        "-c", "copy",
        "-avoid_negative_ts", "1",
        output_path
    ]

    subprocess.run(command, check=True)
    print("8K video cut successfully without quality loss ✅")


cut_8k_video_lossless(
    "VID_20240719_121542_20251016110716.mp4",
    r"D:\Abderrahim Internship Data\output_cut.mp4",
    "00:00:15",
    "00:07:00"
)
