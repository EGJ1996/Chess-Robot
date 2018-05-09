from __future__ import print_function
import re,sys,time
from itertools import count
from collections import OrderedDict,namedtuple
from chess import Chess
from algorithm import Algorithm
import serial
import datetime
import imutils
import time
import cv2
import automaticedge as a
import numpy as np
#import frame_diff as f
#import matlab.engine
ser = serial.Serial('COM3',9600)
#ser.open()
A1, H1, A8, H8 = 91, 98, 21, 285
if sys.version_info[0] == 2:
    input = raw_input
    class NewOrderedDict(OrderedDict):
        def move_to_end(self, key):
            value = self.pop(key)
            self[key] = value
    OrderedDict = NewOrderedDict


min_area = 2000

camera = cv2.VideoCapture(0)
time.sleep(5)
acc = False
corners = [None]*4
cnt = 0
def mousePos(event, x, y, flags, param):
    global cnt
    if event == cv2.EVENT_LBUTTONDOWN and cnt<5:
        #ans = input('Is this a corner')
        #if ans=='y' or ans=='Y':
        corners[cnt] = (x,y)
        cnt+=1
        print('counter is ',cnt)
        print('coordinates ',(x,y))

def detect():
    global corners
    change = False
    firstFrame = None
    finalFrame = None
    for i in range(50):
        (grabbed, f1) = camera.read()
    initialFrame = imutils.resize(f1, width=500)
    (h1,w1) = initialFrame.shape[:2]
    cv2.imshow('initial before loop',initialFrame)
    counter = 0
    while True:
        counter+=1
        (grabbed, frame) = camera.read()
        text = "Unoccupied"

        if not grabbed:
            break

        frame = imutils.resize(frame, width=500)
        #frame = imutils.resize(frame,width=320,height=320)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if firstFrame is None:
            firstFrame = gray
            continue
        if(len(corners)==4 and corners[3]!=None):
            M = cv2.getPerspectiveTransform(np.float32(corners), np.float32([(0, 0), (320, 0), (0, 320), (320, 320)]))
            frame = cv2.warpPerspective(frame, M, (320, 320))
            frameDelta = cv2.absdiff(firstFrame, gray)
            thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

            thresh = cv2.dilate(thresh, None, iterations=2)
            (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)

            for c in cnts:
                if cv2.contourArea(c) < min_area:
                    continue
                (x, y, w, h) = cv2.boundingRect(c)
                (x,y,w,h) = (x-corners[0][0],y-corners[1][1],w,h)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                text = "Occupied"
                change = True

        '''cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)'''

        cv2.imshow("Security Feed", frame)
        cv2.setMouseCallback('Security Feed',mousePos)
        if cnt >4:
            cv2.setMouseCallback('Security Feed',None)
        #cv2.imshow("Thresh", thresh)
        #cv2.imshow("Frame Delta", frameDelta)
        key = cv2.waitKey(1) & 0xFF
        '''if acc==False and counter%50==0:
            temp,corners = a.chess_corners_HSV(initialFrame)
            M = cv2.getPerspectiveTransform(np.float32(corners), np.float32([(0, 0), (320, 0), (0, 320), (320, 320)]))
            temp = cv2.warpPerspective(initialFrame,M, (320, 320))
            cv2.imshow('initFrame',temp)
            time.sleep(3)
            ans = input("Is this the right transformation?")
            if ans=='y' or ans=='Y':
                acc = True
                '''
        if text == "Unoccupied" and change == True:
            time.sleep(3)
            (ret, finalFrame) = camera.read()
            break

        if key == ord("q"):
            break

    '''rows = finalFrame.shape[0]
    cols = finalFrame.shape[1]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -90, 1)
    finalFrame = cv2.warpAffine(finalFrame, M, (cols, rows))
    rows = initialFrame.shape[0]
    cols = initialFrame.shape[1]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -90, 1)
    initialFrame = cv2.warpAffine(initialFrame, M, (cols, rows))
    cv2.imshow('img', initialFrame)
    #cv2.imshow('im', finalFrame)
    print(a.determine_move(initialFrame,finalFrame))
    # finalFrame = adjust_gamma(finalFrame,5)
    finalFrame = a.chess_corners_HSV(finalFrame)
    #initialFrame = a.chess_corners(initialFrame)
    cv2.imshow('final', finalFrame)
    cv2.imshow('img', initialFrame)'''
    
    print('Checkpoint 1')
    cv2.imshow('initial after loop',initialFrame)
    finalFrame = imutils.resize(finalFrame, width=500)
    #initialFrame,corners = a.chess_corners_HSV(initialFrame)
    #finalFrame,corners1 = a.chess_corners_HSV(finalFrame,corners)
    M = cv2.getPerspectiveTransform(np.float32(corners), np.float32([(0, 0), (320, 0), (0, 320), (320, 320)]))
    initialFrame = cv2.warpPerspective(initialFrame, M, (320, 320))
    finalFrame = cv2.warpPerspective(finalFrame, M, (320, 320))
    '''for i in range(4):
        cv2.circle(finalFrame,(corners[i][0],corners[i][1]), 3, (0,0,255), -1)'''
    #finalFrame = a.chess_corners_HSV(finalFrame,corners)[0]
    rows = finalFrame.shape[0]
    cols = finalFrame.shape[1]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -90, 1)
    finalFrame = cv2.warpAffine(finalFrame, M, (cols, rows))
    rows = initialFrame.shape[0]
    cols = initialFrame.shape[1]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -90, 1)
    initialFrame = cv2.warpAffine(initialFrame, M, (cols, rows))
    #cv2.imshow('final',finalFrame)
    '''initialFrame = cv2.cvtColor(initialFrame, cv2.COLOR_BGR2GRAY)
    finalFrame = cv2.cvtColor(finalFrame,cv2.COLOR_BGR2GRAY)
    frameDelta = cv2.absdiff(initialFrame, finalFrame)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)'''
    #cv2.imshow('thresh',thresh)
    rows = thresh.shape[0]
    cols = thresh.shape[1]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -90, 1)
    thresh = cv2.warpAffine(thresh, M, (cols, rows))
    cv2.imshow('initial',initialFrame)
    cv2.imshow('final',finalFrame)
    #cv2.imshow('thresh',thresh)
    '''initial = imutils.resize(initialFrame, width=500)
    initial = cv2.cvtColor(initial, cv2.COLOR_BGR2GRAY)
    initial = cv2.GaussianBlur(initial, (21, 21), 0)
    final = imutils.resize(finalFrame, width=500)
    final = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
    final = cv2.GaussianBlur(final, (21, 21), 0)
    frameDelta = cv2.absdiff(initial, final)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)'''
    '''initialFrame = cv2.cvtColor(initialFrame, cv2.COLOR_BGR2GRAY)
    finalFrame = cv2.cvtColor(finalFrame, cv2.COLOR_BGR2GRAY)
    thresh = cv2.absdiff(initialFrame,finalFrame)'''
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
    initialFrame1 = cv2.cvtColor(initialFrame, cv2.COLOR_BGR2GRAY)
    initialFrame1 = cv2.GaussianBlur(initialFrame1, (21, 21), 0)
    finalFrame1 = cv2.cvtColor(finalFrame, cv2.COLOR_BGR2GRAY)
    finalFrame1 = cv2.GaussianBlur(finalFrame1, (21, 21), 0)
    frameDelta = cv2.absdiff(initialFrame1, finalFrame1)
    initialFrame1 = clahe.apply(initialFrame1)
    finalFrame1 = clahe.apply(finalFrame1)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    #res = a.determine_move4(initialFrame1,finalFrame1)
    #print(a.determine_move3(thresh))
    return a.determine_move3(thresh)
    #corners = None

def parse(c):
    fil, rank = ord(c[0]) - ord('a'), int(c[1]) - 1
    return A1 + fil - 10*rank


def render(i):
    rank, fil = divmod(i - A1, 10)
    return chr(fil + ord('a')) + str(-rank + 1)

def print_pos(pos):
    print()
    uni_pieces = {'r':'♜', 'n':'♞', 'b':'♝', 'q':'♛', 'k':'♚', 'p':'♟',
                  'R':'♖', 'N':'♘', 'B':'♗', 'Q':'♕', 'K':'♔', 'P':'♙', '.':'·'}
    for i, row in enumerate(pos.state.board):
        print(' ',7-i+1, '  '.join(uni_pieces.get(p, p) for p in row))
    print('    a   b   c   d  e   f   g  h \n\n')

def convertMove(mv):
    col  = ord(mv[0]) - ord('a')
    row = int(mv[1])
    return (7-row+1)*8+(7-col)

def main():
    pos = Chess()
    layers=5
    algorithm=Algorithm(pos,'black',layers)
    print_pos(pos)
    cnt1 = 0
    delayTime = 25;
    while True:
        res = detect()
        move = str(res[0])+str(res[1])
        print("move is")
        print(move)
        match=re.match('([a-h][1-8])'*2,move)

        if not match or (pos.format_move(move) not in pos.legal_moves()):
            print('Illegal Move')
            #move=input('Your move:')
            move = str(res[1])+str(res[0])
            match=re.match('([a-h][1-8])'*2,move)
            if not match or (pos.format_move(move) not in pos.legal_moves()):
                continue

        #move=chb.determine_move()
        pos.move(move)
        print_pos(pos)
        opp_move=algorithm.best_move()
        newMove = str(opp_move[0])+str(ord('9')-ord(opp_move[1]))+str(opp_move[2])+ str(ord('9')-ord(opp_move[3]))
        print(newMove)
        
        start = opp_move[0:2]
        end = opp_move[2:]
        a = 63 - convertMove(newMove[0:2])
        b = 63 - convertMove(newMove[2:])
        if(pos.state.board[b//8][b%8]!='· '):
            ser.write((str(convertMove(newMove[0:2]))+","+str(convertMove(newMove[2:]))+"!").encode())
            delayTime = 60        
        else:
            ser.write((str(convertMove(newMove[0:2]))+","+str(convertMove(newMove[2:]))+".").encode())
        pos.move(opp_move)
        print_pos(pos)
        time.sleep(delayTime)
	#return opp_move

def warmup():
    for i in range(0,31,2):
        ser.write((str(i)+","+str(i+1)+".").encode())
        time.sleep(10)
if __name__ == '__main__':
    main()
    #warmup()
