import numpy as  np
import cv2
import math
import matplotlib.pyplot as plt

# calculate z from pdf.
def GuassianGenerator(standardVariance, mean):
  rand = np.random.random()
  plusOrMinus = np.random.random()
  z = 0

  beforLog = rand * standardVariance * (np.sqrt(2*np.pi))
  log = np.log(beforLog)
  if log < 0:
    log = 1
  afterLog = np.sqrt( (2*(standardVariance**2)) / log )
  if plusOrMinus > 0.5:
    afterLog *= -1
  z = mean + afterLog

  return z

def addGuassianNoise(inputImage, standardVariance, mean):
  width = len(inputImage[0])
  height = len(inputImage)

  outputImage = np.zeros((height,width),np.int32)

  for i in xrange(height):
    for j in xrange(width):
      outputImage[i][j] = inputImage[i][j] + GuassianGenerator(standardVariance, mean)
      if outputImage[i][j] > 255:
        outputImage[i][j] = 255
      elif outputImage[i][j] < 0:
        outputImage[i][j] = 0

  return outputImage

def saltAndPepperGenerator(salt, pepper):
  rand = np.random.random()
  noise = 0

  if rand < salt:
    return 255
  elif rand > salt and rand < (pepper + salt):
    return -255
  else:
    return 0

def addSaltAndPepperNoise(inputImage, salt, pepper):
  width = len(inputImage[0])
  height = len(inputImage)

  outputImage = np.zeros((height,width),np.int32)

  for i in xrange(height):
    for j in xrange(width):
      outputImage[i][j] = inputImage[i][j] + saltAndPepperGenerator(salt, pepper)
      if outputImage[i][j] > 255:
        outputImage[i][j] = 255
      elif outputImage[i][j] < 0:
        outputImage[i][j] = 0

  return outputImage

# scaling image
def scaling(inputImage):
    width = len(inputImage[0])
    height = len(inputImage)

    # the interval of the whole image
    mMax = np.amax(inputImage)
    mMin = np.amin(inputImage)

    scaling = np.zeros((height, width), np.float64)

    for i in range(height):
        for j in range(width):
                scaling[i, j] = int((inputImage[i, j] - mMin) * 255 / (mMax - mMin))

    return scaling

def filter2d(inputImage, inputFilter, mode):
  width = len(inputImage[0])
  height = len(inputImage)

  outputImage = np.zeros((height, width), np.float64)
  
  if mode == "arithmetic mean":
    for i in range(height):
      for j in range(width):
        outputImage[i, j] = arithmeticMean(inputImage, inputFilter, i, j, mode)
  elif mode == "geometric mean":
    for i in range(height):
      for j in range(width):
        outputImage[i, j] = geometricMean(inputImage, inputFilter, i, j, mode)
  elif mode == "median":
    for i in range(height):
      for j in range(width):
        outputImage[i, j] = Median(inputImage, inputFilter, i, j, mode)
  elif mode == "harmonic mean":
    for i in range(height):
      for j in range(width):
        outputImage[i, j] = harmonicMean(inputImage, inputFilter, i, j, mode)
  elif mode == "contraharmonic mean":
    for i in range(height):
      for j in range(width):
        outputImage[i, j] = contraharmonicMean(inputImage, inputFilter, i, j, mode, -2)
  elif mode == "min":
    for i in range(height):
      for j in range(width):
        outputImage[i, j] = Median(inputImage, inputFilter, i, j, mode)

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

# calculate according to the formula
def geometricMean(inputImage, inputFilter, x, y, mode):
    filterSize = len(inputFilter)
    sumOfFilter = 0

    for i in range(filterSize):
        for j in range(filterSize):
            sumOfFilter += inputFilter[i][j]

    pointCorrelation = correaltion(inputImage, inputFilter, x, y, mode)
    return int(math.pow(pointCorrelation, 1 / float(sumOfFilter)))

def contraharmonicMean(inputImage, inputFilter, x, y, mode, Q):
  filterSize = len(inputFilter)

  pointCorrelation = correaltion(inputImage, inputFilter, x, y, mode, Q)
  tmpCorrelation = correaltion(inputImage, inputFilter, x, y, mode, Q + 1)
  if pointCorrelation == 0:
    return 0
  else:
    return int(tmpCorrelation / float(pointCorrelation))

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

# calculate according to the formula
def Median(inputImage, inputFilter, x, y, mode):
    width = len(inputImage[0])
    height = len(inputImage)
    filterSize = len(inputFilter)
    movement = int(filterSize / 2)

    sort = np.zeros(filterSize**2, np.int32)

    for i in range(filterSize):
        for j in range(filterSize):
            newX = x - movement + i;
            newY = y - movement + j;

            if (newX < 0) or (newX > height - 1) or (newY < 0) or (newY > width - 1):
                sort[i * filterSize + j] = 0
            else:
            	   sort[i * filterSize + j] = inputImage[newX][newY]

    sort.sort()
    if mode == "min":
      return sort[0]
    elif mode == "median":
      if filterSize**2 % 2 == 0:
        return sort[(filterSize**2) / 2]
      else:
        return int((sort[(filterSize**2) / 2] + sort[(filterSize**2) / 2 - 1]) / float(2))

# calculate correaltion
def correaltion(inputImage, inputFilter, x, y, mode, Q = 1):
    width = len(inputImage[0])
    height = len(inputImage)
    filterSize = len(inputFilter)
    movement = int(filterSize / 2)
    
    pointCorrelation = 0

    if mode == "geometric mean":
      pointCorrelation = 1

    for i in range(filterSize):
        for j in range(filterSize):
            newX = x - movement + i;
            newY = y - movement + j;

            if (newX < 0) or (newX > height - 1) or (newY < 0) or (newY > width - 1):
                pointCorrelation += 0
            elif mode == "arithmetic mean":
                pointCorrelation += inputImage[newX][newY] * inputFilter[i][j]
            elif mode == "geometric mean":
              if  inputImage[newX, newY] == 0:
                pointCorrelation *= 1
              else:
                pointCorrelation *= long(inputImage[newX][newY] * inputFilter[i][j])
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

if __name__ == "__main__":
  inputImage = cv2.imread("./task_2.png", 0)

  standardVariance = 40
  mean = 0
  GuassianImage = addGuassianNoise(inputImage, standardVariance, mean)
  cv2.imwrite("GuassianImage.png", GuassianImage)

  salt = 0.2
  pepper = 0.2
  SaltAndPepperImage = addSaltAndPepperNoise(inputImage, salt, pepper)
  cv2.imwrite("SaltAndPepperImage.png", SaltAndPepperImage)

  salt = 0.2
  pepper = 0
  SaltImage = addSaltAndPepperNoise(inputImage, salt, pepper)
  cv2.imwrite("SaltImage.png", SaltImage)

  threeFilter = [[1 for i in range(3)] for i in range(3)]
  GuassianAriImage = filter2d(GuassianImage, threeFilter, "arithmetic mean")
  cv2.imwrite("GuassianAriImage.png", GuassianAriImage)

  GuassianGeoImage = filter2d(GuassianImage, threeFilter, "geometric mean")
  cv2.imwrite("GuassianGeoImage.png", GuassianGeoImage)

  GuassianMedianImage = filter2d(GuassianImage, threeFilter, "median")
  cv2.imwrite("GuassianMedianImage.png", GuassianMedianImage)

  SaltAndPepperAriImage = filter2d(SaltAndPepperImage, threeFilter, "arithmetic mean")
  cv2.imwrite("SaltAndPepperAriImage.png", SaltAndPepperAriImage)

  SaltAndPepperGeoImage = filter2d(SaltAndPepperImage, threeFilter, "geometric mean")
  cv2.imwrite("SaltAndPepperGeoImage.png", SaltAndPepperGeoImage)

  SaltAndPepperHarImage = filter2d(SaltAndPepperImage, threeFilter, "harmonic mean")
  cv2.imwrite("SaltAndPepperHarImage.png", SaltAndPepperHarImage)

  SaltAndPepperMedianImage = filter2d(SaltAndPepperImage, threeFilter, "median")
  cv2.imwrite("SaltAndPepperMedianImage.png", SaltAndPepperMedianImage)

  SaltMinImage = filter2d(SaltImage, threeFilter, "min")
  cv2.imwrite("SaltMinImage.png", SaltMinImage)

  SaltHarImage = filter2d(SaltImage, threeFilter, "harmonic mean")
  cv2.imwrite("SaltHarImage.png", SaltHarImage)

  SaltConImage = filter2d(SaltImage, threeFilter, "contraharmonic mean")
  cv2.imwrite("SaltConImage.png", SaltConImage)