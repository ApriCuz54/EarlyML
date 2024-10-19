import pygame, sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2

#initialising everything required
WINDOWSIZEX = 640
WINDOWSIZEY = 580

WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)

IMAGESAVE = False

MODEL = load_model("bestmodel.h5")

LABEL = {0:"Zero", 1:"One", 2:"Two", 
         3:"Three", 4:"Four", 5:"Five", 
         6:"Six", 7:"Seven", 8:"Eight", 
         9:"Nine"}

iswriting = False


#initialise pygame
pygame.init()
DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))
FONT = pygame.font.Font("freesansbold.ttf", 20)
pygame.display.set_caption("Digit Board")

number_xcord = []
number_ycord = []
BOUNDRYINC = 5

image_cnt = 1

PREDICT = True

#keeps the pygame running till we click the close button
while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            sys.exit()
        
        if event.type == MOUSEBUTTONDOWN:
            iswriting = True
        
        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (xcord, ycord), 4, 0)

            number_xcord.append(xcord)
            number_ycord.append(ycord)
        

        if event.type == MOUSEBUTTONUP:
            iswriting = False
            number_xcord = sorted(number_xcord)
            number_ycord = sorted(number_ycord)

            rect_min_x, rect_max_x = number_xcord[0]-BOUNDRYINC, number_xcord[-1]+BOUNDRYINC
            rect_min_y, rect_max_y = number_ycord[0]-BOUNDRYINC, number_ycord[-1]+BOUNDRYINC

            number_xcord = []
            number_ycord = []

            img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x, rect_min_y:rect_max_y]
            if IMAGESAVE:
                cv2.imwrite("image.png")
                image_cnt += 1

            if PREDICT:
                img_arr = img_arr.astype(np.float32)
                img_arr = cv2.resize(img_arr, (28,28))
                img_arr = img_arr/255

                pred = MODEL.predict(np.expand_dims(img_arr, axis=0))[0]
                label = LABEL[np.argmax(pred)]

                textSurface = FONT.render(label, True, RED, WHITE)
                textRecObj = textSurface.get_rect()
                textRecObj.topleft = (rect_min_x, rect_min_y)  # Position the label at the top-left corner of the rectangle

                DISPLAYSURF.blit(textSurface, textRecObj)

    pygame.display.flip()
