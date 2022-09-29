import os
from ffmpeg import audio
from pathlib import Path
import parselmouth
from parselmouth.praat import run_file
from pydub import AudioSegment


def AudioAnalysis(dir, file):
    sound = os.path.join(dir, file) 
    dir_path = os.path.dirname(os.path.realpath(__file__))  # current dir 
    source_run = os.path.join(dir_path, "myspsolution.praat")
    try:
        objects = run_file(source_run, -20, 2, 0.27, "yes",sound, dir, 80, 400, 0.01, capture_output=True, return_variables = True)
        # 第四个参数为原praat脚本中的 Minimum_pause_duration（若有bug可适当调小）
        totDur = objects[2]['originaldur']
        nPause = objects[2]['npause']
        arDur = objects[2]['speakingtot']
        nSyl = objects[2]['voicedcount']
        arRate = objects[2]['articulationrate']
    except:
        totDur = 0
        nPause = 0
        arDur = 0
        nSyl = 0
        arRate = 0
        print("Try again the sound of the audio was not clear")
    return round(totDur, 2), int(nPause), round(arDur, 2), int(nSyl), round(arRate, 2)

def FixSpeed(totDur_ori: float, 
             nPause_ori: int, 
             arDur_ori: float, 
             nSyl_ori: int, 
             arRate_ori: float, 
             audio_syn):
    speed_factor = 0
    path_syn, filename_syn = os.path.split(audio_syn)
    name_syn, suffix_syn = os.path.splitext(filename_syn)
    totDur_syn, nPause_syn, arDur_syn, nSyl_syn, arRate_syn = AudioAnalysis(path_syn, filename_syn)

    print(f"for original audio:\n\ttotDur = {totDur_ori}s\n\tnPause = {nPause_ori}\n\tarDur = {arDur_ori}s\n\tnSyl = {nSyl_ori}\n\tarRate = {arRate_ori} per second\n-----")
    print(f"for synthesized audio:\n\ttotDur = {totDur_syn}s\n\tnPause = {nPause_syn}\n\tarDur = {arDur_syn}s\n\tnSyl = {nSyl_syn}\n\tarRate = {arRate_syn} per second\n-----")
    speed_factor = round(arRate_ori/arRate_syn, 2)
    print(f"speed_factor = {speed_factor}")
    if speed_factor == 0:
        print("error!\n The speed factor is 0")
        return audio_syn
    else:
        out_file = os.path.join(path_syn, name_syn + "_{}".format(speed_factor) + suffix_syn)
        audio.a_speed(audio_syn, speed_factor, out_file)
        os.remove(audio_syn)  # remove intermediate wav files
        print(f"Finished!\nThe path of out_file is {out_file}")
    return out_file


def TransFormat(fullpath, out_suffix):
    path_, name = os.path.split(fullpath)
    name, _ = os.path.splitext(name)
    sourcefile = AudioSegment.from_file(fullpath)
    out_file = os.path.join(path_, name + "." + str(out_suffix))  
    sourcefile.export(out_file, format = str(out_suffix))
    return str(out_file)


def DelFile(rootDir, matchText: str):
    fileList = os.listdir(rootDir)
    for file in fileList:
        if matchText in file:
            delFile = os.path.join(rootDir, file)
            os.remove(delFile)
            print("Deleted：", delFile)


def work(totDur_ori: float, 
         nPause_ori: int, 
         arDur_ori: float, 
         nSyl_ori: int, 
         arRate_ori: float, 
         audio_syn):
    fix_file = FixSpeed(totDur_ori, 
                        nPause_ori, 
                        arDur_ori, 
                        nSyl_ori, 
                        arRate_ori, 
                        audio_syn)
    # DelFile(in_path, '.TextGrid')
    out_path, _ = os.path.split(audio_syn)
    DelFile(out_path, '.TextGrid')
    return fix_file


