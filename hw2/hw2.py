import numpy as np
import cv2
import matplotlib.pyplot as plt

def equalize_hist(inputImage):

    width = len(inputImage[0])
    height = len(inputImage)

    # initialize 2 arrays with zero
    calculateHist = [0 for i in range(256)]
    newHist = [0 for i in range(256)]

    # calculate the histogram of inputImage
    for j in range(height):
        for k in range(width):
            calculateHist[inputImage[j, k]] += 1

    # equalize histogram according to the formula
    # L - 1 = 256 - 1 = 255, MN = width * height
    for i in range(256):
        newHist[i] = 255 * sum([calculateHist[j] for j in range(i+1)]) / (width * height)

    # new the image with utf-8, otherwise something wrong will happen
    outputImage = np.zeros((height, width), np.int32)

    # assign image
    for m in range(height):
        for n in range(width):
            outputImage[m, n] = newHist[inputImage[m, n]]

    return outputImage

def filter2d(inputImage, inputFilter, mode):

    width = len(inputImage[0])
    height = len(inputImage)

    outputImage = np.zeros((height, width), np.int32)

    # weighted mean filter
    if mode == "smooth":
        for i in range(height):
            for j in range(width):
                outputImage[i, j] = averageFilter(inputImage, inputFilter, i, j)
    # sharpen filter
    elif mode == "sharpen":
        sharpImage = np.zeros((height, width), np.int32)

        for i in range(height):
            for j in range(width):
                outputImage[i, j] = inputImage[i][j] - laplacianFilter(inputImage, LaplacianFilter, i, j)
        outputImage = scaling(outputImage)

        # #  the final effect of image
        # for i in range(height):
        #     for j in range(width):
        #         sharpImage[i, j] = outputImage[i, j] + inputImage[i][j]

        # sharpImage = scaling(sharpImage)
        # cv2.imwrite("sharpImage.png", sharpImage)

    # high boost filter
    elif mode == "boost":
        outputImage = boostFilter(inputImage)

    return outputImage

# calculate according to the formula
def averageFilter(inputImage, inputFilter, x, y):
    filterSize = len(inputFilter)
    sumOfFilter = 0
    for i in range(filterSize):
        for j in range(filterSize):
            sumOfFilter += inputFilter[i][j]

    pointCorrelation = conrrealtion(inputImage, inputFilter, x, y)
    return int(pointCorrelation / sumOfFilter)

# calculate according to the formula
def laplacianFilter(inputImage, LaplacianFilter, x, y):
    return conrrealtion(inputImage, LaplacianFilter, x, y)

# calculate according to the formula
def boostFilter(inputImage):
    width = len(inputImage[0])
    height = len(inputImage)
    fuzzyImage = filter2d(inputImage, threeOrderFilter, "smooth")

    gmaskImage = np.zeros((height, width), np.int32)

    for i in range(height):
        for j in range(width):
            gmaskImage[i][j] = inputImage[i][j] - fuzzyImage[i][j]

    outputImage = np.zeros((height, width), np.int32)
    factor = 4
    for i in range(height):
        for j in range(width):
            outputImage[i][j] = inputImage[i][j] + factor * gmaskImage[i][j]

    return outputImage

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

def forFun(inputImage):

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

    return outputImage

def forFunWeight(inputImage, tmp):

    width = len(inputImage[0])
    height = len(inputImage)

    # initialize 6 arrays with zero, devided by 2 groups
    calculateHist1 = [0 for i in range(256)]

    newHist1 = [0 for i in range(256)]

    # calculate the histogram of inputImage
    for j in range(height):
        for k in range(width):
            calculateHist1[inputImage[j, k][tmp]] += 1

    # equalize histogram according to the formula
    # L - 1 = 256 - 1 = 255, MN = width * height
    for i in range(256):
        newHist1[i] = 255 * sum([calculateHist1[j] for j in range(i+1)]) / (width * height)

    # new the image with utf-8, otherwise something wrong will happen
    outputImage = np.zeros((height,width, 3),np.uint8)

    # assign image
    for m in range(height):
        for n in range(width):
            outputImage[m, n][tmp] = newHist1[inputImage[m, n][tmp]]

    return outputImage

# print histogram graph
def printHistForFun(inputImage, name):
    width = len(inputImage[0])
    height = len(inputImage)

    calculateHist1 = [0 for i in range(256)]
    calculateHist2 = [0 for i in range(256)]
    calculateHist3 = [0 for i in range(256)]

    for j in range(height):
        for k in range(width):
            calculateHist1[inputImage[j, k][0]] += 1
            calculateHist2[inputImage[j, k][1]] += 1
            calculateHist3[inputImage[j, k][2]] += 1

    plt.bar(np.arange(256), calculateHist1, color = 'g', edgecolor = 'g')
    plt.savefig(name + "Blue.png")

    plt.bar(np.arange(256), calculateHist2, color = 'g', edgecolor = 'g')
    plt.savefig(name + "Green.png")

    plt.bar(np.arange(256), calculateHist3, color = 'g', edgecolor = 'g')
    plt.savefig(name + "Red.png")

if __name__ == "__main__":
    inputImage = cv2.imread("./74.png", 0)

    # equalize histogram
    printHist(inputImage, "inputImageHistogram.png")

    outputImage = equalize_hist(inputImage)
    cv2.imwrite("equalizeHist.png", outputImage)
    printHist(outputImage, "outputImageHistogram.png")

    outputImageAgain = equalize_hist(cv2.imread("./equalizeHist.png", 0))
    cv2.imwrite("equalizeHistAgain.png", outputImageAgain)
    printHist(outputImageAgain, "outputImageHistogramAgain.png")

    # filter
    threeOrderFilter = [[1 for i in range(3)] for i in range(3)]
    threeOrderImage = filter2d(inputImage, threeOrderFilter, "smooth")
    printHist(threeOrderImage, "threeOrderImageHistogram.png")
    cv2.imwrite("threeOrderImage.png", threeOrderImage)

    sevenOrderFilter = [[1 for i in range(7)] for i in range(7)]
    sevenOrderImage = filter2d(inputImage, sevenOrderFilter, "smooth")
    printHist(sevenOrderImage, "sevenOrderImageHistogram.png")
    cv2.imwrite("sevenOrderImage.png", sevenOrderImage)

    elevenOrderFilter = [[1 for i in range(11)] for i in range(11)]
    elevenOrderImage = filter2d(inputImage, elevenOrderFilter, "smooth")
    printHist(elevenOrderImage, "elevenOrderImageHistogram.png")
    cv2.imwrite("elevenOrderImage.png", elevenOrderImage)

    LaplacianFilter = [[1,1,1], [1,-8,1], [1,1,1]]
    LaplacianImage = filter2d(inputImage, LaplacianFilter, "sharpen")
    cv2.imwrite("LaplacianImage.png", LaplacianImage)
    LaplacianAgainImage = cv2.imread("./LaplacianImage.png", 0)
    printHist(LaplacianAgainImage, "LaplacianHistogram.png")

    boostImage = filter2d(inputImage, [], "boost")
    cv2.imwrite("boostImage.png", boostImage)
    boostAgainImage = cv2.imread("./boostImage.png", 0)
    printHist(boostAgainImage, "boostHistogram.png")

    #extra
    forFunImage = cv2.imread("./forFun.jpg", 1)
    printHistForFun(forFunImage, "ImageHistogramForFun")
    outputForFunImage = forFun(forFunImage)
    printHistForFun(outputForFunImage, "outputImageHistogramForFun")
    cv2.imwrite("equalizeHistForFun.png", outputForFunImage)

    outputForFunImageRed = forFunWeight(forFunImage, 2)
    cv2.imwrite("equalizeHistForFunRed.png", outputForFunImageRed)

    outputForFunImageGreen = forFunWeight(forFunImage, 1)
    cv2.imwrite("equalizeHistForFunGreen.png", outputForFunImageGreen)

    outputForFunImageBlue = forFunWeight(forFunImage, 0)
    cv2.imwrite("equalizeHistForFunBlue.png", outputForFunImageBlue)
    
