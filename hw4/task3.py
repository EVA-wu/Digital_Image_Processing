import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

def histogram(inputImage):

    width = len(inputImage[0])
    height = len(inputImage)

    # initialize 6 arrays with zero, devided by 2 groups
    calculateHist1 = [0 for i in range(256)]
    calculateHist2 = [0 for i in range(256)]
    calculateHist3 = [0 for i in range(256)]

    newHist1 = [0 for i in range(256)]
    newHist2 = [0 for i in range(256)]
    newHist3 = [0 for i in range(256)]

    # calculate the histogram of inputImage
    for j in range(height):
        for k in range(width):
            calculateHist1[inputImage[j, k][0]] += 1
            calculateHist2[inputImage[j, k][1]] += 1
            calculateHist3[inputImage[j, k][2]] += 1

    # equalize histogram according to the formula
    # L - 1 = 256 - 1 = 255, MN = width * height
    for i in range(256):
        newHist1[i] = 255 * sum([calculateHist1[j] for j in range(i+1)]) / (width * height)
        newHist2[i] = 255 * sum([calculateHist2[j] for j in range(i+1)]) / (width * height)
        newHist3[i] = 255 * sum([calculateHist3[j] for j in range(i+1)]) / (width * height)

    # new the image with utf-8, otherwise something wrong will happen
    outputImage = np.zeros((height,width, 3),np.uint8)

    # assign image
    for m in range(height):
        for n in range(width):
            outputImage[m, n][0] = newHist1[inputImage[m, n][0]]
            outputImage[m, n][1] = newHist2[inputImage[m, n][1]]
            outputImage[m, n][2] = newHist3[inputImage[m, n][2]]

    return (outputImage, newHist1, newHist2, newHist3)

def averageHist(histRed, histGreen, histBlue):
  averHist = [0 for i in range(256)]
  for i in range(len(histRed)):
    averHist[i] = (histRed[i] + histGreen[i] + histBlue[i]) / 3

  return averHist

def buildFromAvarageHist(inputImage):
  width = len(inputImage[0])
  height = len(inputImage)

  histImage, histRed, histGreen, histBlue = histogram(inputImage)
  histAverage = averageHist(histRed, histGreen, histBlue)
  
  outputHist = [0 for i in range(256)]

  for i in range(256):
    outputHist[i] = 255 * sum([histAverage[j] for j in range(i+1)]) / (width * height)

  outputImage = np.zeros((height, width, 3), np.int32)

  for m in range(height):
        for n in range(width):
            outputImage[m, n][0] = outputHist[inputImage[m, n][0]]
            outputImage[m, n][1] = outputHist[inputImage[m, n][1]]
            outputImage[m, n][2] = outputHist[inputImage[m, n][2]]

  return outputImage 

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

# print histogram graph
def printHist(inputImage, name):
    width = len(inputImage[0])
    height = len(inputImage)

    calculateHist = [0 for i in range(256)]

    for j in range(height):
        for k in range(width):
            calculateHist[inputImage[j, k]] += 1

    plt.bar(np.arange(256), calculateHist, color = 'g', edgecolor = 'g')
    plt.savefig(name)

if __name__=="__main__":
  inputImage = cv2.imread("./74.png", 1)

  histImage, histRed, histGreen, histBlue = histogram(inputImage)
  cv2.imwrite("hist.png", histImage)

  averageHistImage = buildFromAvarageHist(inputImage)
  cv2.imwrite("averageHistImage.png", averageHistImage)

  RGB2HSI(inputImage)
  intImage = cv2.imread("./intImage.png")
  histIntImage, histIntRed, histIntGreen, histIntBlue = histogram(inputImage)
  cv2.imwrite("histIntImage.png", histIntImage)
