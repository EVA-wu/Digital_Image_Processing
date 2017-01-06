# -*- coding:utf-8 -*-

import numpy as np
import cv2

def equalize_hist(img):
    height = len(img)
    width  = len(img[0])

    hist = calculate_hist(img)

    fx = [0 for i in range(256)]

    for i in range(256):
        fx[i] = 255 * sum([hist[j] for j in range(i+1)]) / (width * height)

    new_img = np.zeros((height,width),np.uint8)

    for x in range(height):
        for y in range(width):
                new_img[x, y] = fx[img[x, y]]

    return new_img


def calculate_hist(img):
    hist = [0 for i in range(256)]

    height = len(img)
    width  = len(img[0])
    
    for x in range(height):
        for y in range(width):
            hist[img[x,y]] += 1

    return hist

if __name__ == '__main__':
    img = cv2.imread("./Mean_Filtered_image.png", 0)

    outImage = equalize_hist(img)
    cv2.imwrite("outImage.png", outImage)

