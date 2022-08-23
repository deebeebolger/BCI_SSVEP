# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 10:10:36 2019

@author: ALU
"""
import threading
import pygame
import time
from math import sin, pi
import sys

NUM_OF_THREAD = 2
b = threading.Barrier(NUM_OF_THREAD)

def blinking_block(points, frequency):
    COUNT = 1
    CLOCK = pygame.time.Clock()
    ''' FrameRate '''
    FrameRate = 60

    b.wait()  # Synchronize the start of each thread
    while True:  # execution block
        CLOCK.tick(FrameRate)
        tmp = sin(2 * pi * frequency * (COUNT / FrameRate))
        color = 255 * (tmp > 0)
        print(f"Color is {color}\n")
        #block = pygame.draw.polygon(win, (color, color, color), points, 0)
        block = pygame.draw.circle(win, (0, color, 0), points,200)
        pygame.display.update(block)  # can't update in main thread which will introduce delay in different block
        COUNT += 1
        print(f"Count is: {COUNT}\n")
        if COUNT == FrameRate:
            COUNT = 0
        print(CLOCK.get_time()) #check the time between each frame (144HZ=7ms; 60HZ=16.67ms)


if __name__ == '__main__':
    pygame.init()
    pygame.TIMER_RESOLUTION = 1  # set time resolutions
    win = pygame.display.set_mode((1800, 1000))

    # background canvas
    bg = pygame.Surface(win.get_size())
    bg = bg.convert()
    bg.fill((0, 0, 0))  # black background
    # display
    win.blit(bg, (0, 0))
    pygame.display.update()
    pygame.display.set_caption("Blinking")

    ''' frequency '''
    #frequency = [8, 9, 10, 11, 12, 13]  # frequency bank
    frequency = [12, 8]
    ''' POINTS '''
    POINTS = [[300, 400], [1400, 400]]


    duration = 10  # in seconds

    threads = []
    for i in range(len(POINTS)):
        threads.append(threading.Thread(target=blinking_block, args=(POINTS[i], frequency[i])))
        threads[i].setDaemon(True)
        threads[i].start()

    timer_event = pygame.USEREVENT+1
    pygame.time.set_timer(timer_event, int(duration)*1000)
    RUN = True
    while RUN:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                RUN = False
                pygame.quit()
            elif event.type == timer_event:
                RUN = False
                pygame.quit()
        pygame.time.delay(1)

exit()