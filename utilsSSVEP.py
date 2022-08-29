from screeninfo import get_monitors
import pygame
from pygame.locals import *
import os
import sys
from flickit import Flick
import time
from record_data import RecordData
from live_recorder import LiveRecorder
import joblib
import numpy as np
from preprocess import preprocess_recordings
from subprocess import Popen

pygame.init()

def time_str():
    return time.strftime("%H_%M_%d_%m_%Y", time.gmtime())


def render_waiting_screen(text_string=None, time_black=0.0):
    pygame.init()
    # pygame.font.init()
    h_str = str(get_monitors()[0])
    w, h = (h_str.split(',')[2], h_str.split(',')[3])
    hstr = int(float(h.split('=')[1]))
    wstr = int(float(w.split('=')[1]))
    display_x = int(float(wstr)/2)
    display_y = int(float(hstr)/2)
    display_x, display_y = (2 * display_x, 2 * display_y)
    os.environ['SDL_VIDEO_CENTERED'] = '1'
    window = pygame.display.set_mode((display_x, display_y), pygame.NOFRAME, 32)
    pygame.display.set_caption("SSVEP")
    if time_black > 0:
        window.fill((0., 0., 0.))
        timer_event = USEREVENT + 1
        pygame.time.set_timer(timer_event, int(time_black)*1000)
        myfont = pygame.font.SysFont("arial", 40)
    else:
        myfont = pygame.font.SysFont("arial", 50)
        press_string = "Please press the Space Bar to continue..."
        textsurface1 = myfont.render(press_string, False, (0, 0, 0))
        text_rect1 = textsurface1.get_rect(center=(display_x/2, display_y/2+100))
        if text_string:
            textsurface2 = myfont.render(text_string, False, (0, 0, 0))
            text_rect2 = textsurface2.get_rect(center=(display_x/2, display_y/2-100))
        window.fill((150, 100, 150))
        window.blit(textsurface1, text_rect1)
        if text_string:
            window.blit(textsurface2, text_rect2)
    pygame.display.update()
    busy = True
    while busy:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
                busy = False
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    exit()
                    busy = False
                    return False
                elif event.key == K_SPACE:
                    press_string1 = ""
                    textsurface_new = myfont.render(press_string1, False, (0, 0, 0))
                    window1 = pygame.display.set_mode((display_x, display_y), pygame.NOFRAME, 32)
                    window1.fill((0, 0, 0))
                    window1.blit(textsurface_new, text_rect1)
                    pygame.quit()
                    busy = False
                    return False

            if not (time_black > 0.):
                window.blit(textsurface1, text_rect1)
                if text_string:
                    window.blit(textsurface2, text_rect2)
            else:
                if event.type == timer_event:
                    pygame.quit()


                    return False
            if busy == False:
                break
            pygame.display.update()


def begin_experiment_1(freq, trials=5):
    if not os.path.isdir("REC"):
        os.mkdir("REC")

    render_waiting_screen("Welcome to this SSVEP experiment")
    render_waiting_screen("The experiment will start now...There will be 5 trials of 15seconds.")
    render_waiting_screen("A 2second long black screen will appear between each trial.")
    recorder = RecordData(250., 20.)           # First arg = Fs, second arg = participant age.
    render_waiting_screen(text_string=None, time_black=2.0)
    recorder.start_recording()

    for i in range(0, int(trials)):

        recorder.add_trial(int(freq))
        Flick(float(freq)).flicker(15.)
        tdata = recorder.add_trial(0.)
        recorder.freqdetect_fbcca(tdata, freq)    # Call of RecordData class freqdete_fbcca to apply filter-bank CCA.
        render_waiting_screen(text_string=None, time_black=2.0)

    filename = "REC/%s_freq_%s.mat" % (time_str(), freq)
    recorder.stop_recording_and_dump(filename)
    recorder.killswitch.terminate = True
    recorder = None

    render_waiting_screen("That was the last one, thank you for participation!")
    sys.exit()


def begin_experiment_2(str_list):
    #display_x, display_y = get_display_resolution()
    for m in get_monitors():
        print(str(m))
    display_x, display_y = (m.width, m.height)
    display_x, display_y = (2 * display_x, 2 * display_y)
    os.environ['SDL_VIDEO_CENTERED'] = '1'
    window = pygame.display.set_mode((display_x, display_y), pygame.NOFRAME, 32)
    pygame.display.set_caption("SSVEP")
    window.fill((0, 0, 0))
    pygame.display.update()

    if os.name == 'nt':
    	for command in str_list:
            command_parts = command.split(" ")
            #print("start /d "+command)
            #os.system("start /d "+command)
            Popen(command_parts)
    elif os.name == 'posix':
        os.system("|".join(str_list))
    else:
        print("Could not get OS-name!")




if __name__ == "__main__":
    begin_experiment_1()
    exit()
