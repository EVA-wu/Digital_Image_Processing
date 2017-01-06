import numpy as  np
import cv2
import math
import matplotlib.pyplot as plt

def init():
  width = 64
  height = 64

  outputImage = np.zeros((height,width, 3),np.uint8)

  for i in range(height):
    for j in range(width):
      if i < 32 and j < 32:
        outputImage[i, j][0] = 255
        outputImage[i, j][1] = 255
      elif i > 31 and j > 31:
        outputImage[i, j][1] = 255
      elif i >31 and j < 32:
        outputImage[i, j][1] = 255
        outputImage[i, j][2] = 255 
      elif i < 32 and j > 31:
        outputImage[i, j][0] = 255
        outputImage[i, j][2] = 255

  cv2.imwrite("init.png", outputImage)

def RGB2HSI(inputImage):
  width = len(inputImage[0])
  height = len(inputImage)

  hueImage = np.zeros((height,width, 3),np.uint8)
  saImage = np.zeros((height,width, 3),np.uint8)
  intImage = np.zeros((height,width, 3),np.uint8)

  for i in range(height):
    for j in range(width):
      R = inputImage[i, j][0]
      G = inputImage[i, j][1]
      B = inputImage[i, j][2]
      num = ((R - G) + (R - B)) / 2
      den = math.sqrt(math.pow(R - G, 2) + (R - B) * (G - B))
      ceta = np.arccos(num / den)
      if G >= B:
        hueImage[i, j] = ceta
      elif G < B:
        hueImage[i, j] = 360 - ceta
      intImage[i, j] = (R + G + B) / 3
      saImage[i, j] = 1 - 3 * min(R, G, B) / (R + G + B)

  cv2.imwrite("intImage.png", intImage)
  cv2.imwrite("saImage.png", saImage)
  cv2.imwrite("hueImage.png", hueImage)

def HSI2RGBWithBlur(hueImage, saImage, intImage):
  width = len(hueImage[0])
  height = len(hueImage)

  outputImage = np.zeros((height,width, 3),np.uint8)

  # if blurImage == "Saturation":
  #   saImage = blurImage(saImage)
  # elif blurImage == "Hue":
  hueImage = blurImage(hueImage)

  for i in range(height):
    for j in range(width):
      hueImage[i, j] *= 360
      I = intImage[i, j]
      S = saImage[i, j]
      H = hueImage[i, j]
      if hueImage[i, j] < 120 and hueImage[i, j] >= 0:
        B = I * (1 - S)
        R = I * (1 + (S * math.cos(H)) / math.cos(60 - H) )
        G = 3 * I - (R + B)
      elif hueImage[i, j] < 240 and hueImage[i, j] >= 120:
        H -= 120
        R = I * (1 - S)
        G = I * (1 + (S * math.cos(H)) / math.cos(60 - H) )
        B = 3 * I - (B + G)
      elif hueImage[i, j] < 360 and hueImage[i, j] >= 240:
        H -= 240
        G = I * (1 - S)
        B = I * (1 + (S * math.cos(H)) / math.cos(60 - H) )
        R = 3 * I - (G + B)
      outputImage[i, j][0] = R
      outputImage[i, j][1] = G     
      outputImage[i, j][2] = B
  cv2.imwrite("resultHue.png", outputImage)          

def blurImage(inputImage):
    width = len(inputImage[0])
    height = len(inputImage[1])
    outputImage = np.zeros((height, width), np.int32)

    sixteenOrderFilter = [[1 for i in range(16)] for i in range(16)]
    for i in range(height):
            for j in range(width):
                outputImage[i, j] = averageFilter(inputImage, sixteenOrderFilter, i, j)

    outputImage = scaling(outputImage)
    return outputImage

# calculate according to the formula
def averageFilter(inputImage, inputFilter, x, y):
    filterSize = len(inputFilter)
    sumOfFilter = 0

    print inputFilter
    for i in range(filterSize):
        for j in range(filterSize):
            sumOfFilter += inputFilter[i][j]

    pointCorrelation = conrrealtion(inputImage, inputFilter, x, y)
    print pointCorrelation
    return int(pointCorrelation / sumOfFilter)

# calculate correaltion
def conrrealtion(inputImage, inputFilter, x, y):
    width = len(inputImage[0])
    height = len(inputImage)
    filterSize = len(inputFilter)
    movement = int(filterSize / 2)

    pointCorrelation = 0

    for i in range(filterSize):
        for j in range(filterSize):
            newX = x - movement + i;
            newY = y - movement + j;

            if (newX < 0) or (newX > height - 1) or (newY < 0) or (newY > width - 1):
                pointCorrelation += 0
            else:
                pointCorrelation += inputImage[newX][newY] * inputFilter[i][j]

    return pointCorrelation

# scaling image
def scaling(inputImage):
    width = len(inputImage[0])
    height = len(inputImage)

    # the interval of the whole image
    mMax = np.amax(inputImage)
    mMin = np.amin(inputImage)

    scaling = np.zeros((height, width), np.int32)

    for i in range(height):
        for j in range(width):
                scaling[i, j] = int((inputImage[i, j] - mMin) * 255 / (mMax - mMin))

    return scaling

if __name__ == "__main__":
    init()
    RGB2HSI(cv2.imread("init.png", 1))
    HSI2RGBWithBlur(cv2.imread("hueImage.png", 0), cv2.imread("saImage.png", 0), cv2.imread("intImage.png", 0))