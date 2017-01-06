import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import Image

def scalePre(inputImage, level):
    width = int(np.round(len(inputImage[0]) * level))
    height = int(np.round(len(inputImage) * level))

    return scale(inputImage, (width, height))

def scale(input_img, size):
    temp = Image.fromarray(input_img)

    width = len(input_img[0])
    height = len(input_img)

    # result image
    result = np.zeros((size[1], size[0]), np.float64)

    # scale factors
    scaleX = float(size[0])/width
    scaleY = float(size[1])/height

    # expand the initial image to ensure the points like (0, 0) has N4
    tmp = np.zeros((height + 2, width + 2), np.float64)
    for i in range(len(tmp)):
        for j in range(len(tmp[0])):
            if j != 0 and j != len(tmp[0]) - 1 and i != 0 and i != len(tmp) - 1:
                    tmp[i][j] = input_img[i - 1][j - 1]
            else:
                    tmp[i][j] = 1

    # find the position of (ri, rj) in tmp and use find its pixel in result by bi-linear interpolation
    for ri in range(len(result)):
        for rj in range(len(result[0])):
            # if ri or rj is equal to 0, the position will be out of range
            if  ri != 0 and rj != 0:
                   # according to the equation
                ii = float(ri - 1) / scaleX
                jj = float(rj - 1) / scaleY

                i = int(ii)
                j = int(jj)

                u = ii - i
                v = jj - j

                i = i + 1
                j = j + 1
                # bi-linear for interpolation
                result[ri][rj] = (1 - u) * (1 - v) * tmp[i][j] + (1 - u) * v * tmp[i][j + 1] + u * (1 - v) * tmp[i + 1][j] + u * v * tmp[i + 1][j + 1]

    return result

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

def fastGuidance(inputImage, guidanceImage):
    p = scalePre(inputImage, 1)
    I = scalePre(guidanceImage, 1)

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

    a = scale(a, (len(guidanceImage[0]), len(guidanceImage)) )
    b = scale(b, (len(guidanceImage[0]), len(guidanceImage)) )

    q = np.multiply(guidanceImage, a) + b

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
    
    q1 = fastGuidance(pImage1, IImage1)
    q2 = fastGuidance(pImage2, IImage2)
    q3 = fastGuidance(pImage3, IImage3)

    for i in range(height):
        for j in range(width):
            q[i][j][0] = q1[i][j]
            q[i][j][1] = q2[i][j]
            q[i][j][2] = q3[i][j]

    cv2.imwrite("./dataset/img_enhancement/tulipsFastGuide.bmp", q)
