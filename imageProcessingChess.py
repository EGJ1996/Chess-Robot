import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from enum import Enum
import imutils
import os
import time
border_size=45
gama=1.7
size_x=65
size_y=48
roi_sz=20
black_piece=[0,0,0]
white_piece=[0,0,0]
white_square=[0,0,0]
black_square=[0,0,0]
prev=[0]*64
nxt=[0]*64
corners = None


class Chess(Enum):
    BLACK_SQUARE=1
    WHITE_SQUARE=2
    BLACK_PIECE=3
    WHITE_PIECE=4
def Euclidean_distance(v1,v2):
    return math.sqrt((v1[0]-v2[0])**2+(v1[1]-v2[1])**2+(v1[2]-v2[2])**2)

x_kern = np.arange(0, 4, 1, float)
y_kern = x_kern[:,np.newaxis]
x_kern0 = y_kern0 = 4 // 2
OPENING_KERNEL = np.uint8(np.exp(-4*np.log(2) * ((x_kern-x_kern0)**2 + (y_kern-y_kern0)**2) / 2**2) * 255)

def chess_corners_HSV(image):
    global corners
    lower_red = np.array([30, 100, 150])
    upper_red = np.array([40, 225, 255])

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(image, image, mask=mask)

    #cv2.imshow('frame', image)
    #cv2.imshow('mask', mask)
    #cv2.imshow('res', res)

    cor = cv2.findNonZero(mask)

    lu = (10000, 10000)
    ru = (0, 10000)
    ld = (10000, 0)
    rd = (0, 0)
    minSum = 1000
    maxSum = 0
    minDiff = 1000
    maxDiff = -1000
    for i in range(len(cor)):
        if cor[i][0][0] + cor[i][0][1] < minSum:
            lu = cor[i][0]
            minSum = cor[i][0][0] + cor[i][0][1]
        if cor[i][0][0] + cor[i][0][1] > maxSum:
            rd = cor[i][0]
            maxSum = cor[i][0][0] + cor[i][0][1]
        if cor[i][0][0] - cor[i][0][1] > maxDiff:
            ru = cor[i][0]
            maxDiff = cor[i][0][0] - cor[i][0][1]
        if cor[i][0][0] - cor[i][0][1] < minDiff:
            ld = cor[i][0]
            minDiff = cor[i][0][0] - cor[i][0][1]

    if corners == None:
        corners = []
        corners.append(lu)
        corners.append(ru)
        corners.append(ld)
        corners.append(rd)
    '''cv2.circle(frame, tuple(lu), 3, (0, 0, 0), -1)
    cv2.circle(frame, tuple(ru), 3, (255, 255, 255), -1)
    cv2.circle(frame, tuple(ld), 3, (0, 0, 255), -1)
    cv2.circle(frame, tuple(rd), 3, (255, 0, 0), -1)'''

    #cv2.imshow("final", image)

    M = cv2.getPerspectiveTransform(np.float32([lu, ru, ld, rd]), np.float32([(0, 0), (320, 0), (0, 320), (320, 320)]))
    image = cv2.warpPerspective(image, M, (320, 320))

    # cv2.imshow('first transform',image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 35, 60)
    # cv2.imshow('Second Canny',edged)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(edged, kernel)
    # cv2.imshow('Second dilated',dilated)
    # _, cnts, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    (_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    print(len(cnts))
    screenCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.07 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break
    # cv2.drawContours(image, cnts, 0, (0, 255, 0), 3)
    minSum = 1000
    maxSum = 0
    minDiff = 1000
    maxDiff = -1000
    for i in range(len(cnts[0])):
        if cnts[0][i][0][0] + cnts[0][i][0][1] < minSum:
            lu = cnts[0][i][0]
            minSum = cnts[0][i][0][0] + cnts[0][i][0][1]
        if cnts[0][i][0][0] + cnts[0][i][0][1] > maxSum:
            rd = cnts[0][i][0]
            maxSum = cnts[0][i][0][0] + cnts[0][i][0][1]
        if cnts[0][i][0][0] - cnts[0][i][0][1] > maxDiff:
            ru = cnts[0][i][0]
            maxDiff = cnts[0][i][0][0] - cnts[0][i][0][1]
        if cnts[0][i][0][0] - cnts[0][i][0][1] < minDiff:
            ld = cnts[0][i][0]
            minDiff = cnts[0][i][0][0] - cnts[0][i][0][1]

    '''cv2.circle(image,tuple(lu), 3, (0,0,0), -1)
    cv2.circle(image,tuple(ru),3, (255,255,255), -1)
    cv2.circle(image,tuple(ld), 3, (0,0,255), -1)
    cv2.circle(image,tuple(rd), 3, (255,0,0), -1)'''
    # cv2.imshow('corn',image)
    #M = cv2.getPerspectiveTransform(np.float32([lu, ru, ld, rd]), np.float32([(0, 0), (320, 0), (0, 320), (320, 320)]))
    # image = cv2.warpPerspective(image,M,(320,320))
    #cv2.imshow("dilated", dilated)
    #cv2.imshow('Second transform', image)
    return image





def chess_corners(image):
    image = cv2.blur(image,(3,3))
    ratio = image.shape[0] / 300.0
    orig = image.copy()
    image = imutils.resize(image, height = 300)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #gray = cv2.equalizeHist(gray)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 50, 70)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    dilated = cv2.dilate(edged, kernel)
    #cv2.imshow('dil',dilated)
    _, cnts, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #(_,cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
    screenCnt = None
    for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.07 * peri, True)
            if len(approx) == 4:
                    screenCnt = approx
                    break

    #cv2.drawContours(image, cnts, 0, (0, 255, 0), 3)
    #cv2.imshow('Image before transform',image)
    lu = (10000,10000)
    ru = (0,10000)
    ld = (10000,0)
    rd = (0,0)
    print('first contour')
    '''for i in range(len(cnts[0])):
            if cnts[0][i][0][0] <= lu[0] and cnts[0][i][0][1] <= lu[1]:
                lu = cnts[0][i][0]
            if cnts[0][i][0][0] >= ru[0] and cnts[0][i][0][1] <= ru[1]:
                ru = cnts[0][i][0]
            if cnts[0][i][0][0] <= ld[0]+3 and cnts[0][i][0][1] >= ld[1]:
                ld = cnts[0][i][0]
            if cnts[0][i][0][0] >= rd[0] and cnts[0][i][0][1] >= rd[1]:
                rd = cnts[0][i][0]'''

    minSum = 1000
    maxSum = 0
    minDiff = 1000
    maxDiff = -1000
    for i in range(len(cnts[0])):
            if cnts[0][i][0][0] + cnts[0][i][0][1] < minSum:
                    lu = cnts[0][i][0]
                    minSum = cnts[0][i][0][0] + cnts[0][i][0][1]
            if cnts[0][i][0][0] + cnts[0][i][0][1] > maxSum:
                    rd = cnts[0][i][0]
                    maxSum = cnts[0][i][0][0] + cnts[0][i][0][1]
            if cnts[0][i][0][0] - cnts[0][i][0][1] > maxDiff:
                    ru = cnts[0][i][0]
                    maxDiff = cnts[0][i][0][0] - cnts[0][i][0][1]
            if cnts[0][i][0][0] - cnts[0][i][0][1] < minDiff:
                    ld = cnts[0][i][0]
                    minDiff = cnts[0][i][0][0] - cnts[0][i][0][1]
                   

    '''for i in range(len(cnts[0])):
            cv2.circle(image,tuple(cnts[0][i][0]),3,(0,0,255),-1)'''
    '''cv2.circle(image,tuple(cnts[0][0][0]),3,(0,0,255),-1)
    cv2.circle(image,tuple(cnts[0][len(cnts[0])//2][0]),3,(0,0,0),-1)
    cv2.circle(image,tuple(cnts[0][len(cnts)-1][0]),3,(255,255,255),-1)'''
    #t2=tuple(cnts[0][len(cnts[0])-2][0])
    '''cv2.circle(image,tuple(lu), 3, (0,0,0), -1)
    cv2.circle(image,tuple(ru),3, (255,255,255), -1)
    cv2.circle(image,tuple(ld), 3, (0,0,255), -1)
    cv2.circle(image,tuple(rd), 3, (255,0,0), -1)'''
    #cv2.imshow("dilated", dilated)
    M = cv2.getPerspectiveTransform(np.float32([lu,ru,ld,rd]),np.float32([(0,0),(320,0),(0,320),(320,320)]))
    image = cv2.warpPerspective(image,M,(320,320))

    #cv2.imshow('first transform',image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 35, 60)
    #cv2.imshow('Second Canny',edged)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    dilated = cv2.dilate(edged, kernel)
    #cv2.imshow('Second dilated',dilated)
    #_, cnts, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    (_,cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
    print(len(cnts))
    screenCnt = None
    for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.07 * peri, True)
            if len(approx) == 4:
                    screenCnt = approx
                    break
    #cv2.drawContours(image, cnts, 0, (0, 255, 0), 3)
    minSum = 1000
    maxSum = 0
    minDiff = 1000
    maxDiff = -1000
    for i in range(len(cnts[0])):
            if cnts[0][i][0][0] + cnts[0][i][0][1] < minSum:
                    lu = cnts[0][i][0]
                    minSum = cnts[0][i][0][0] + cnts[0][i][0][1]
            if cnts[0][i][0][0] + cnts[0][i][0][1] > maxSum:
                    rd = cnts[0][i][0]
                    maxSum = cnts[0][i][0][0] + cnts[0][i][0][1]
            if cnts[0][i][0][0] - cnts[0][i][0][1] > maxDiff:
                    ru = cnts[0][i][0]
                    maxDiff = cnts[0][i][0][0] - cnts[0][i][0][1]
            if cnts[0][i][0][0] - cnts[0][i][0][1] < minDiff:
                    ld = cnts[0][i][0]
                    minDiff = cnts[0][i][0][0] - cnts[0][i][0][1]

    '''cv2.circle(image,tuple(lu), 3, (0,0,0), -1)
    cv2.circle(image,tuple(ru),3, (255,255,255), -1)
    cv2.circle(image,tuple(ld), 3, (0,0,255), -1)
    cv2.circle(image,tuple(rd), 3, (255,0,0), -1)'''
    #cv2.imshow('corn',image)
    M = cv2.getPerspectiveTransform(np.float32([lu,ru,ld,rd]),np.float32([(0,0),(320,0),(0,320),(320,320)]))
    #image = cv2.warpPerspective(image,M,(320,320))
    #cv2.imshow("dilated", dilated)
    #cv2.imshow('Second transform',image)
    return image


def mask(img, lower, upper):
    #hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    #cv2.imshow('hsv',hsv)
    mask = cv2.inRange(img, lower, upper)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, OPENING_KERNEL)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, OPENING_KERNEL)
    return closing

BLACK_LOWER = (0,0,0)
BLACK_UPPER = (120,120,120)
def black_mask(img):
    return mask(img, BLACK_LOWER, BLACK_UPPER)


def dist(t1,t2):
    return math.sqrt((t1[0]-t2[0])**2+(t1[1]-t2[1])**2)

def detect_state(img):
    #img=cv2.imread(im)
    #img=imutils.resize(im,width=640,height=480)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img,50,150,apertureSize = 3)
    minLineLength=100
    img = chess_corners_HSV(img)
    #cv2.circle(img,(5,315), 3, (0,255,0), -1)
    lu = (0,0)
    ru = (320,0)
    ld = (0,320)
    rd = (320,320)
    x_d = (ru[0] - lu[0])//8
    y_d = (ld[1] - lu[1])//8
    squares=[]
    curY = lu[1]
    for i in range(8):
        curX=lu[0]
        for j in range(8):
            squares.append([(curX,curY),(curX+x_d,curY),(curX+x_d,curY+y_d),(curX,curY+y_d)])
            curX=curX+x_d
        curY=curY+y_d


    for s in squares:
        for i in range(len(s)):
            cv2.circle(img,tuple(s[i]), 3, (0,0,255), -1)
    
    cv2.imshow('image',img)
        
    blur=cv2.GaussianBlur(img,(5,5),0)
    final=np.zeros(img.shape)
    _,binar=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    binar = cv2.Canny(img,100,120)
    #_,conts,hier=cv2.findContours(binar,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(final, conts, -1, (0,255,0), 3)
    kernel = np.ones((7,7),np.uint8)
    dil = cv2.dilate(binar,kernel,iterations = 1)
    sumWhite=[0]*64
    cv2.imshow('dilated',dil)
    for i in range(64):
        s=squares[i]
        midY=s[0][1]+int((s[3][1]-s[0][1])/2)
        midX=s[0][0]+int((s[1][0]-s[0][0])/2)
        for y in range(midY-15,midY+15):
            for x in range(midX-15,midX+15):
                sumWhite[i]=sumWhite[i]+dil[y][x]
        
    '''print('sum of whites is')
    for i in range(64):
        print(str(i)+'--->'+str(sumWhite[i]))'''
    taken=[False]*64
    for i in range(64):
        if sumWhite[i]<100000:
            taken[i]=False
        else:
            taken[i]=True

    #cv2.imshow('image',img)
    return (img,taken,squares)

def detect_squares():
    chess=[0]*64
    for i in range(0,64,16):
        for j in range(i,i+8):
            chess[j]=j%2
    for i in range(8,64,16):
        for j in range(i,i+8):
            chess[j]=(j+1)%2
    return chess
    
def detect_black(img,squares):
    img=black_mask(img)
    sumBlack=[0]*64
    isBlack=[False]*64
    for i in range(64):
        s=squares[i]
        for y in range(s[0][1]+7,s[2][1]-7):
            for x in range(s[0][0]+7,s[1][0]-7):
                sumBlack[i]=sumBlack[i]+img[y][x]

    for i in range(0,64):
        #print(sumBlack[i])
        if sumBlack[i]>=100000:
            isBlack[i]=True
    return isBlack
    
def determine_move(cur,final):
    move=''
    first=''
    second=''
    #cur = cv2.imread('pawn2.jpg')
    #final = cv2.imread('knight3.jpg')
    tup=detect_state(cur)
    img1=tup[0]
    sq=tup[2]
    img2=detect_state(final)[0]
    #cv2.imshow('image1',img1)
    #cv2.imshow('image2',img2)
    blur=cv2.GaussianBlur(img1,(5,5),0)
    gray=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    final=np.zeros(img1.shape)
    _,binar=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    binar = cv2.Canny(img1,100,120)
    kernel = np.ones((3,3),np.uint8)
    #dil1 = cv2.dilate(binar,kernel,iterations = 1)
    blur=cv2.GaussianBlur(img2,(5,5),0)
    gray=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    final=np.zeros(img2.shape)
    _,binar=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    binar = cv2.Canny(img2,100,120)
    #_,conts,hier=cv2.findContours(binar,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(final, conts, -1, (0,255,0), 3)
    kernel = np.ones((3,3),np.uint8)
    dil2 = cv2.dilate(binar,kernel,iterations = 1)
    cv2.imshow('dil1',dil1)
    cv2.imshow('dil2',dil2)
    frame_diff=cv2.absdiff(dil1,dil2)
    cv2.imshow('Diff',frame_diff)
    sum_diff=[0]*64
    for i in range(64):
        s=sq[i]
        for y in range(s[0][1]+10,s[2][1]-10):
            for x in range(s[0][0]+10,s[1][0]-10):
                    sum_diff[i]=sum_diff[i]+frame_diff[y][x]

    ind1=0
    ind2=0
    s_diff=sum_diff[:]
    sort=sorted(s_diff)
    max1=sort[-1]
    max2=sort[-2]
    for i in range(64):
        if sum_diff[i]==max1:
            ind1=i
        elif sum_diff[i]==max2:
            ind2=i

    print(ind1)
    print(ind2)
    first=str(chr(ord('a')+ind1%8)+str(8-(ind1//8)))
    second=str(chr(ord('a')+ind2%8)+str(8-(ind2//8)))
    s1=sq[ind1]
    s2=sq[ind2]
    sum1=0
    sum2=0
    for y in range(s1[0][1],s1[2][1]):
        for x in range(s1[0][0],s1[1][0]):
            sum1=sum1+dil1[y][x]
    for y in range(s2[0][1],s2[2][1]):
        for x in range(s2[0][0],s2[1][0]):
            sum2=sum2+dil2[y][x]

    if sum1>sum2:
        move=first+second
    else:
        move=second+first
    return move

'''def determine_move2(thresh):
    

    M = cv2.getPerspectiveTransform(np.float32([corners[0], corners[1], corners[2], corners[3]]), np.float32([(0, 0), (320, 0), (0, 320), (320, 320)]))
    thresh = cv2.warpPerspective(thresh, M, (320, 320))
    cv2.imshow("thresh", thresh)
    lu = (0,0)
    ru = (320,0)
    ld = (0,320)
    rd = (320,320)
    x_d = (ru[0] - lu[0])//8
    y_d = (ld[1] - lu[1])//8
    squares = []
    curY=lu[1]
    for i in range(8):
        curX= lu[0]
        for j in range(8):
            squares.append([(curX,curY),(curX+x_d,curY),(curX+x_d,curY+y_d),(curX,curY+y_d)])
            curX=curX+x_d
        curY=curY+y_d
    sum_diff=[0]*64
    for i in range(64):
        s=squares[i]
        print (s)
        for y in range(s[0][1],s[2][1]):
            for x in range(s[0][0],s[1][0]):
                    sum_diff[i]=sum_diff[i]+thresh[y][x]

    ind1=0
    ind2=0
    s_diff=sum_diff[:]
    sort=sorted(s_diff)
    max1=sort[-1]
    max2=sort[-2]
    for i in range(64):
        if sum_diff[i]==max1:
            ind1=i
        elif sum_diff[i]==max2:
            ind2=i

    return (ind1,ind2)'''

def determine_move2(cur,final):
    img1 = chess_corners_HSV(cur)
    img2 = chess_corners_HSV(final)
    lu = (0,0)
    ru = (320,0)
    ld = (0,320)
    rd = (320,320)
    x_d = (ru[0] - lu[0])//8
    y_d = (ld[1] - lu[1])//8
    squares = []
    curY=lu[1]
    for i in range(8):
        curX= lu[0]
        for j in range(8):
            squares.append([(curX,curY),(curX+x_d,curY),(curX+x_d,curY+y_d),(curX,curY+y_d)])
            curX=curX+x_d
        curY=curY+y_d
    blur=cv2.GaussianBlur(img1,(5,5),0)
    gray=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    final=np.zeros(img1.shape)
    _,binar=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    binar1 = cv2.Canny(img1,100,120)
    kernel = np.ones((3,3),np.uint8)
    dil1 = cv2.dilate(binar,kernel,iterations = 1)
    blur=cv2.GaussianBlur(img2,(5,5),0)
    gray=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    final=np.zeros(img2.shape)
    _,binar=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    binar2 = cv2.Canny(img2,100,120)
    _,conts,hier=cv2.findContours(binar,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(final, conts, -1, (0,255,0), 3)
    kernel = np.ones((3,3),np.uint8)
    dil2 = cv2.dilate(binar,kernel,iterations = 1)
    #cv2.imshow('dil1',dil1)
    #cv2.imshow('dil2',dil2)
    frame_diff=cv2.absdiff(dil1,dil2)
    #cv2.imshow('Diff',frame_diff)
    sum_diff = [0]*64
    for i in range(64):
        s=squares[i]
        for y in range(s[0][1],s[2][1]):
            for x in range(s[0][0],s[1][0]):
                    sum_diff[i]=sum_diff[i]+frame_diff[y][x]

    ind1=0
    ind2=0
    s_diff=sum_diff[:]
    sort=sorted(s_diff)
    max1=sort[-1]
    max2=sort[-2]
    for i in range(64):
        if sum_diff[i]==max1:
            ind1 = i
        elif sum_diff[i]==max2:
            ind2 = i
    '''dilArr1 = [0]*64
    dilArr2 = [0]*64
    for i in range(64):
        s = squares[i]
        for y in range(s[0][1],s[3][1]):
            for x in range(s[0][0],s[1][0]):
                dilArr1[i] += dil1[y][x]

    for i in range(64):
        s = squares[i]
        for y in range(s[0][1],s[3][1]):
            for x in range(s[0][0],s[1][0]):
                dilArr2[i] += dil2[y][x]

    print('printing dil1')
    for i in range(64):
        print(dilArr1[i])

    print('printing dil2')
    for i in range(64):
        print(dilArr2[i])'''
    move=''
    first=''
    second=''
    first=str(chr(ord('a')+ind1%8)+str(8-(ind1//8)))
    second=str(chr(ord('a')+ind2%8)+str(8-(ind2//8)))
    s1=squares[ind1]
    s2=squares[ind2]
    sum1 = 0
    sum2 = 0
    for y in range(s1[0][1],s1[2][1]):
        for x in range(s1[0][0],s1[1][0]):
            sum1 = sum1 + dil1[y][x]
            sum2 = sum2 + dil2[y][x] 
    '''for y in range(s2[0][1],s2[2][1]):
        for x in range(s2[0][0],s2[1][0]):
            sum2=sum2+dil2[y][x]'''

    
    ''' if sum1>sum2:
        move=second+first
    else:
        move=first+second
    return move'''
    return(second,first)
    #return (ind1,ind2)

    

def test():
    img = cv2.imread('cam1.jpg')
    img = chess_corners(img)
    #detect_state(img)
    
#test()
#print(determine_move())
