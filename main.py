from tkinter import *
from constants import SYSTEM_NAME
import twitter_data_handler
import questionnaire
import subprocess
import cnn


def open_twitter_feed():
    twitter_feed = Toplevel(window)
    twitter_feed.title("Twitter Feed for past 2 weeks")
    # twitter_feed.geometry("400x300")

    for x in twitter_data_handler.get_tweets():
        Label(twitter_feed, text=x.text, height="2", font=("Calibri", 14)).pack()
    Button(twitter_feed, text="Analyse", height="2", width="30", command=analyse_tweets).pack(side=TOP)

def train():
    cnn.run_model(True)


def open_questionnaire():
    questionnaire.SubWindow(window)


def analyse_tweets():
    cnn.run_model(False)
    # cmd = 'python cnn.py models/4cnn-08-0.341-0.849-0.498-0.768.hdf5'
    # p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # out, err = p.communicate()
    # result = out.decode()
    # print("Result : "+result)


window = Tk()
window.title(SYSTEM_NAME)

Label(window, text=SYSTEM_NAME, height="2", font=("Calibri", 15)).pack()

frame_top = Frame(window)
Button(frame_top, text="Connect Twitter Account ", height="2", width="30", command=open_twitter_feed).pack(side=LEFT)
Button(frame_top, text="Questionnaire List", height="2", width="30", command=open_questionnaire).pack(side=LEFT)
Button(frame_top, text="Alert", height="2", width="30", command=train).pack(side=LEFT)
frame_top.pack(side=TOP, fill=BOTH, expand=1)

Button(window, text="Predict", height="2", width="30").pack()

window.mainloop()
