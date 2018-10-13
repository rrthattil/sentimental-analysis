#!/usr/bin/env python
import os,subprocess,time
import pyaudio,wave
import cv2
import Tkinter as tk
from pygame import mixer
from Tkinter import *
from espeak import espeak
mixer.init()
#os.chdir("src/")
def record():
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 7
    WAVE_OUTPUT_FILENAME = "../data/temp/voice.wav"
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,rate=RATE,input=True,frames_per_buffer=CHUNK)
    frames = [] 
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    audio.terminate()
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()
def capture():
    cam = cv2.VideoCapture(0)
    ret,im = cam.read()
    cv2.imwrite("../data/temp/face.png",im)
def face():
    espeak.synth("Smile Please")
    capture()
    f=subprocess.call(['./face-svm-predict'])
    if f==0:
        espeak.synth("you seem angry")
    elif f==1:
        espeak.synth("why are you disgusted")
    elif f==2:
        espeak.synth("You seem happy")
    elif f==3:
        espeak.synth("You have a dead face")
    elif f==4:
        espeak.synth("You seem sad")
    elif f==5:
        espeak.synth("You seem surprised")        
    else:
        espeak.synth("Can't get a read from your face please try again")
def voice():
    espeak.synth("How was your day today")
    time.sleep(4)    
    record()
    v=subprocess.call("./voice-svm-predict")
    if v==0:
        espeak.synth("Well that is a good reason to be mad")
    elif v==1:
        espeak.synth("I will try not to disgust you next")
    elif v==2:
        espeak.synth("Thats nice to hear")
    elif v==3:
        espeak.synth("You sound as dead as a fish")
    elif v==4:
        espeak.synth("Well I am sorry about that")
    elif v==5:
        espeak.synth("No wonder why you are excited")
    elif v==6:
        espeak.synth("Do not fear I am here")        
    else:
        espeak.synth("Can't get a read from your voice please try again")
def analyse():
    if f==v:
        if f==0 or f==1 or f==4 or v==6:
            mixer.music.load('../data/songs/1.mp3')
            mixer.music.play()
        elif f==2 or f==5:
            mixer.music.load('../data/songs/2.mp3')
            mixer.music.play()
        else:
            mixer.music.load('../data/songs/3.mp3')
            mixer.music.play()
    else:
        mixer.music.load('../data/songs/4.mp3')
        mixer.music.play()
root = tk.Tk()
frame = tk.Frame(root)
frame.pack()
b1=tk.Button(frame,text="Analyse Face",command=face)
b1.pack(side=tk.TOP)
b2=tk.Button(frame,text="Analyse Voice",command=voice)
b2.pack(side=tk.TOP)
b3=tk.Button(frame,text="Evaluate Result",command=analyse)
b3.pack(side=tk.TOP)
root.eval('tk::PlaceWindow %s center' % root.winfo_pathname(root.winfo_id()))
root.title("Emotion Detection using  Face & Voice Analysis")
root.geometry("200x100")
root.mainloop()
