import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import Image

def boxFilter(inputImage, r):
    width = len(inputImage[0])
    height = len(inputImage)

    outputImage = np.zeros((height, width), np.float64)
    for i in range(height):
        for j in range(width):
            outputImage[i][j] = inputImage[i][j]
    window = 2 * r + 1
    absW = window**2

    for i in range(height):
        for j in range(width):
        	sum = 0.0
        	outsideWindow = 0
        	for x in range(i - r, i + r + 1):
        	    for y in range(j - r, j + r + 1):
        	        if x < 0 or y < 0 or x > height - 1 or y > width - 1:
        	            outsideWindow = outsideWindow + 1
        	            continue
        	        sum = sum + inputImage[x][y]
        	outputImage[i][j] = sum / (absW - outsideWindow)
    return outputImage 

def guidance(inputImage, guidanceImage):
    p = np.array(inputImage, dtype=float)
    I = np.array(guidanceImage, dtype=float)

    r = 4
    epsilon = 500

    height = len(p)
    width = len(p[0])

    meanP = meanI = boxFilter(I, r)
    meanII = boxFilter(np.multiply(I, I), r)
    meanIP = boxFilter(np.multiply(I, p), r)

    covIP = meanIP - np.multiply(meanI, meanP)
    varI = meanII - np.multiply(meanI, meanI)

    a = boxFilter(covIP / (varI + epsilon), r)
    b = boxFilter(meanP - np.multiply(a, meanI), r)

    q = np.multiply(I, a) + b

    for i in range(height):
        for j in range(width):
        	if q[i][j] < 0:
        	    q[i][j] = 0
        	elif q[i][j] > 255:
        	    q[i][j] = 255
        	else:
        	    q[i][j] = np.floor(q[i][j])

    return q

if __name__=="__main__":
    # inputImage p is identical to guidance I
    pImage = IImage = cv2.imread("./dataset/img_enhancement/tulips.bmp", 1)

    height = len(pImage)
    width = len(pImage[0])

    q = np.zeros((height,width, 3), np.float64)
    
    q1 = np.zeros((height, width), np.float64)
    q2 = np.zeros((height, width), np.float64)
    q3 = np.zeros((height, width), np.float64)

    pImage1 = np.zeros((height, width), np.float64)
    pImage2 = np.zeros((height, width), np.float64)
    pImage3 = np.zeros((height, width), np.float64)

    IImage1 = np.zeros((height, width), np.float64)
    IImage2 = np.zeros((height, width), np.float64)
    IImage3 = np.zeros((height, width), np.float64)

    for i in range(height):
        for j in range(width):
            pImage1[i][j] = pImage[i][j][0]
            pImage2[i][j] = pImage[i][j][1]
            pImage3[i][j] = pImage[i][j][2]
            IImage1[i][j] = IImage[i][j][0]
            IImage2[i][j] = IImage[i][j][1]
            IImage3[i][j] = IImage[i][j][2]
    
    q1 = guidance(pImage1, IImage1)
    q2 = guidance(pImage2, IImage2)
    q3 = guidance(pImage3, IImage3)

    for i in range(height):
        for j in range(width):
            q[i][j][0] = q1[i][j]
            q[i][j][1] = q2[i][j]
            q[i][j][2] = q3[i][j]

    cv2.imwrite("./dataset/img_enhancement/tulipsGuide.bmp", q)
