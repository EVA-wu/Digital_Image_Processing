import cv2
import numpy as np
from basic import *

import FFT

def calculate_pow2(x):
    res = 2
    while res < x:
        res = res << 1
    return res

def filter2d_freq(img, the_filter):
    height = len(img)
    width = len(img[0])
    adjust_height = calculate_pow2(height)
    adjust_width  = calculate_pow2(width)

    FFT_filter = np.zeros((adjust_height,adjust_width), np.int32)
    adjust_img = np.zeros((adjust_height,adjust_width), np.int32)

    for x in xrange(height):
        for y in xrange(width):
            adjust_img[x, y] = img[x, y]

    for x in xrange(len(the_filter)):
        for y in xrange(len(the_filter[0])):
            FFT_filter[x, y] = the_filter[x, y]

    FFT_filter = FFT.centralize(FFT_filter)
    adjust_img = FFT.centralize(adjust_img)

    FFT_filter = FFT.fft2d(FFT_filter, 1)
    FFT_image = FFT.fft2d(adjust_img, 1)

    print "done Fourier transform"

    Result = FFT_image * FFT_filter

    print "done frequecy processing"

    Result = FFT.fft2d(Result, -1).real

    

    print "done Inverse Fourier transform"

    Result_img = np.zeros((height, width), np.float64)

    Result = diolog_transform(Result)

    for x in xrange(height):
        for y in xrange(width):
            Result_img[x, y] = Result[x, y]

    Result_img = centralize(Result_img)

    return Result_img

def mean_filter():
    img = cv2.imread("./74.png", 0)
    the_filter = np.array([[1 for i in xrange(7)] for i in xrange(7)])
    Result = filter2d_freq(img, the_filter)
    
    Result = Fourier_scaling(Result, "log")
    cv2.imwrite("2.png", Result)
    Result = gama_correction(Result, 4)
    cv2.imwrite("Mean_Filtered_image.png", Result)

def laplacian_filter():
    img = cv2.imread("./74.png", 0)
    the_filter = np.array([[1,1,1],[1,-8,1],[1,1,1]])
    Result = filter2d_freq(img, the_filter)
    Result = Fourier_scaling(Result, "log")

    for x in xrange(len(img)):
        for y in xrange(len(img[0])):
            Result[x, y] = 5 * img[x, y] - Result[x, y]

    Result = Fourier_scaling(Result, "log")
    Result = gama_correction(Result, 4)

    cv2.imwrite("Laplacian_Filtered_image.png", Result)

def main():
    mean_filter()
    laplacian_filter()

if __name__ == '__main__':
    main()
