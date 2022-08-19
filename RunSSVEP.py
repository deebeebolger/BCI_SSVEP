import os
import sys
from flickit import Flick
from utilsSSVEP import *
from numpy import mean, var, random
from screeninfo import get_monitors
from AppKit import NSScreen


if __name__ == "__main__":
    args = sys.argv

    # Need to calculate the flickering frequency.
    mon_nom = []  # Name of monitor initialize list.
    fps = []  # Initialize Frames per second list.
    for each in NSScreen.screens():
        print(f"{each.localizedName()}: {each.maximumFramesPerSecond()}Hz")
        fps.append(each.maximumFramesPerSecond())
        mon_nom.append(each.localizedName())

    if str(args[1]) == "exp1":

        begin_experiment_1(args[2])

    elif str(args[1]) == "exp2":
        print("length of args: ", len(args))

        if len(args) == 2:  #Neither stim frequency nor duration specified.

            print(f"The monitor refresh rates are {fps[0]}Hz and {fps[1]}Hz\n")
            fps_pick = input("Enter one of the refresh rates: ")
            fps_pick = int(fps_pick)

            divrs = [3, 4, 6, 8]
            divrs_sel = random.choice(divrs)
            flickfreq = fps_pick/divrs_sel
            print(f"The current flickering frequency is {flickfreq}Hz")

            render_waiting_screen(text_string=None, time_black=0)
            flick_dur = input("Enter duration of flicker stimulus in seconds: ")
            Flick(float(flickfreq)).flicker(flick_dur)

        elif len(args) == 3:  #Flickering frequency specified.
            render_waiting_screen(text_string=None, time_black=0)

            if len(fps)==1:
                print(f"The defined flickering frequency is {args[2]}Hz for a refresh rate of {fps[0]} ")
            elif len(fps)>1:
                for fpsindx in fps:
                    print(f"The defined flickering frequency is {args[2]}Hz for a refresh rate of {fpsindx}\n")

            flick_dur = input("Enter duration of flicker stimulus in seconds: ")
            Flick(float(args[2])).flicker(flick_dur)
        elif len(args) == 4:  #Both flickering frequency and duration specified.

            Flick(float(args[2])).flicker(args[3])
    elif str(args[1]) == "classify":
        start_live_classifier()
    else:
        print("Please specifiy exp1 or exp2 in the first argument!")