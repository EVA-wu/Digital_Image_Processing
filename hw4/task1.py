import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

def filter2d(inputImage, inputFilter, mode):
  width = len(inputImage[0])
  height = len(inputImage)

  outputImage = np.zeros((height, width), np.int32)
  
  if mode == "arithmetic mean":
    for i in range(height):
      for j in range(width):
        outputImage[i, j] = arithmeticMean(inputImage, inputFilter, i, j, mode)
  elif mode == "harmonic mean":
    for i in range(height):
      for j in range(width):
        outputImage[i, j] = harmonicMean(inputImage, inputFilter, i, j, mode)
  elif mode == "contraharmonic mean":
    for i in range(height):
      for j in range(width):
        outputImage[i, j] = contraharmonicMean(inputImage, inputFilter, i, j, mode, 1.5)

  return scaling(outputImage)

# calculate according to the formula
def arithmeticMean(inputImage, inputFilter, x, y, mode):
    filterSize = len(inputFilter)
    sumOfFilter = 0

    for i in range(filterSize):
        for j in range(filterSize):
            sumOfFilter += inputFilter[i][j]

    pointCorrelation = correaltion(inputImage, inputFilter, x, y, mode)
    return int(pointCorrelation / sumOfFilter)

def harmonicMean(inputImage, inputFilter, x, y, mode):
  filterSize = len(inputFilter)
  sumOfFilter = 0

  for i in range(filterSize):
    for j in range(filterSize):
      sumOfFilter += inputFilter[i][j]

  pointCorrelation = correaltion(inputImage, inputFilter, x, y, mode)
  if pointCorrelation == 0:
    return 0
  else:
    return int(sumOfFilter / float(pointCorrelation))

def contraharmonicMean(inputImage, inputFilter, x, y, mode, Q):
  filterSize = len(inputFilter)

  pointCorrelation = correaltion(inputImage, inputFilter, x, y, mode, Q)
  tmpCorrelation = correaltion(inputImage, inputFilter, x, y, mode, Q + 1)
  if pointCorrelation == 0:
    return 0
  else:
    return int(tmpCorrelation / float(pointCorrelation))


# calculate correaltion
def correaltion(inputImage, inputFilter, x, y, mode, Q = 1):
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
            elif mode == "arithmetic mean":
                pointCorrelation += inputImage[newX][newY] * inputFilter[i][j]
            elif mode == "harmonic mean":
                if inputImage[newX, newY] == 0:
                  pointCorrelation +=1
                else:
                  pointCorrelation += 1 / float(inputImage[newX][newY] * inputFilter[i][j])
            elif mode == "contraharmonic mean":
                if inputImage[newX, newY] == 0:
                  pointCorrelation += 1
                else:
                  pointCorrelation += math.pow(inputImage[newX][newY] * inputFilter[i][j], Q) 
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
  inputImage = cv2.imread("task_1.png", 0)
  
  threeFilter = [[1 for i in range(3)] for i in range(3)]
  nineFilter = [[1 for i in range(9)] for i in range(9)]

  threeAriMean = filter2d(inputImage, threeFilter, "arithmetic mean")
  cv2.imwrite("threeAriMean.png", threeAriMean)

  nineAriMean = filter2d(inputImage, nineFilter, "arithmetic mean")
  cv2.imwrite("nineAriMean.png", nineAriMean)

  threeHarMean = filter2d(inputImage, threeFilter, "harmonic mean")
  cv2.imwrite("threeHarMean.png", threeAriMean)

  nineHarMean = filter2d(inputImage, nineFilter, "harmonic mean")
  cv2.imwrite("nineHarMean.png", nineHarMean)

  threeConMean = filter2d(inputImage, threeFilter, "contraharmonic mean")
  cv2.imwrite("threeConMean.png", threeConMean)

  nineConMean = filter2d(inputImage, nineFilter, "contraharmonic mean")
  cv2.imwrite("nineConMean.png", nineConMean)