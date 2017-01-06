import numpy as np
import cv2

#centralize image by formula f(x, y) * (-1)^(x + y)
def centralize(inputImage):
    width = len(inputImage[0])
    height = len(inputImage)

    outputImage = np.zeros((height,width),np.int32)

    for i in range(height):
        for j in range(width):
        	if (i + j) % 2 != 0:
        	    outputImage[i][j] = -1 * inputImage[i][j];
        	else:
        	    outputImage[i][j] = inputImage[i][j]

    return outputImage

# identify flag
def dft2d(inputImage, flag):
    if flag == "DFT":
        return DFT(inputImage)
    elif flag == "IDFT":
        # the inputImage here is the one that has already done DFT
        width = len(inputImage[0])
        height = len(inputImage)
        for x in range(height):
            for y in range(width):
                inputImage[x][y] = inputImage[x][y] / float(width * height)
        return np.conjugate(IDFT(inputImage))

#Inverse Discrete Fourier Transform
def IDFT(inputImage):
    width = len(inputImage[0])
    height = len(inputImage)

    tmpImage = np.zeros((height,width),np.complex256)
    outputImage = np.zeros((height,width),np.complex256)

    for x in range(height):
        for y in range(width):
            for v in range(width):
                tmpImage[x][y] += inputImage[x, v] * np.exp(1j * 2 * np.pi * (float(y*v) / width))

    for x in range(height):
        for y in range(width):
            for u in range(height):
                outputImage[x][y] += tmpImage[u, y] * np.exp(1j * 2 * np.pi * (float(x*u) / height))

    return outputImage

#Discrete Fourier Transform
def DFT(inputImage):
    width = len(inputImage[0])
    height = len(inputImage)

    tmpImage = np.zeros((height,width),np.complex256)
    outputImage = np.zeros((height,width),np.complex256)

    for u in range(height):
        for v in range(width):
            for y in range(width):
                tmpImage[u][v] += inputImage[u, y] * np.exp(-1j * 2 * np.pi * (float(v*y) / width))

    for u in range(height):
        for v in range(width):
            for x in range(height):
                outputImage[u][v] += tmpImage[x, v] * np.exp(-1j * 2 * np.pi * (float(u*x) / height))

    return outputImage

# Log Transform
def logTransform(inputImage):
    width = len(inputImage[0])
    height = len(inputImage)

    outputImage = np.zeros((height,width), np.float64)

    for i in range(height):
        for j in range(width):
        	if inputImage[i][j] > 0:
        	    outputImage[i][j] = np.log(inputImage[i][j])

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

def powerSpace(x):
    res = 2
    while res < x:
        res = res << 1
    return res

def filter2d(inputImage, spatialFilter):
    width = len(inputImage[0])
    height = len(inputImage)

    frequencyImageHeight = powerSpace(height)
    frequencyImageWidth = powerSpace(width)

    row = len(spatialFilter)
    column = len(spatialFilter[0])

    frequencyFilter = np.zeros((frequencyImageHeight, frequencyImageWidth), np.int32)
    frequencyImage = np.zeros((frequencyImageHeight, frequencyImageWidth), np.int32)

    for i in range(height):
        for j in range(width):
            frequencyImage[i][j] = inputImage[i][j]

    for i in range(row):
        for j in range(column):
            frequencyFilter[i][j] = spatialFilter[i][j]

    frequencyImage = centralize(frequencyImage)
    frequencyFilter = centralize(frequencyFilter)

    frequencyImage = dft2d(frequencyImage, "DFT")
    frequencyFilter = dft2d(frequencyFilter, "DFT")

    result = frequencyImage * frequencyFilter

    outputImage = np.zeros((height, width), np.float64)

    result = np.conjugate(dft2d(result, "DFT"))
    result = result.real
    result = rotateTransform(result)

    for x in xrange(height):
        for y in xrange(width):
            outputImage[x][y] = result[x][y]

    return outputImage

# scaling complex
def scalingComplex(inputImage):
    width = len(inputImage[0])
    height = len(inputImage)

    # the interval of the whole image
    mMax = np.amax(inputImage)
    mMin = np.amin(inputImage)
    C= 255 / np.log10(256)

    scaling = np.zeros((height, width), np.int32)

    for i in range(height):
        for j in range(width):
                scaling[i, j] = int(C * np.log10(1 + np.abs(float(255*inputImage[i][j]) / mMax)))

    return scaling

def gama_correction(img, scale):
    table = [0 for i in range(256)]
    for i in range(256):
        val = pow(float(i)/255.0 ,scale) * 255.0
        if val>255:
                val = 255
        elif val<0:
                val = 0
        table[i]= val

    height = len(img)
    width  = len(img[0])

    new_img = np.zeros((height,width), np.uint8)

    for i in range(height):
        for j in range(width):
            if new_img[i, j] < 255:
                new_img[i, j] = int(table[img[i,j]])
            else:
                new_img[i, j] = int(table[255])

    return new_img

#rotate
def rotateTransform(inputImage):
    height, width = inputImage.shape
    outputImage = np.zeros((height,width), np.int32)
    for x in xrange(height):
        for y in xrange(width):
            outputImage[height - x - 1, width - y - 1] = inputImage[x, y]
    return outputImage

if __name__ == '__main__':

    inputImage = cv2.imread("./74.png", 0)

    # dftImage = centralize(inputImage)
    # dftImage = dft2d(dftImage, "DFT")

    # #Inverse Fourier Transform
    # idftImage = dft2d(dftImage, "IDFT")
    # idftImage = idftImage.real
    # idftImage = centralize(idftImage)
    # idftImage = scaling(idftImage)
    # cv2.imwrite("InverseFourierTransform.png", idftImage)
    

    # # Fourier Transform
    # dftImage = np.abs(dftImage)
    # dftImage = logTransform(dftImage)
    # dftImage = scaling(dftImage)
    # cv2.imwrite("FourierTransform.png", dftImage)

    # # Filter in Frequency Field
    # # Mean Filter == Low Pass Filter
    # sevenOrderFilter = [[1 for i in range(7)] for i in range(7)]
    # sevenOrderImage = filter2d(inputImage, sevenOrderFilter)
    # sevenOrderImage = centralize(sevenOrderImage)
    
    
    # sevenOrderImage = scalingComplex(sevenOrderImage)
    # sevenOrderImage = gama_correction(sevenOrderImage, 6)
    # cv2.imwrite("sevenOrderImageDFT.png", sevenOrderImage)

    # Laplacian Filter == High Pass Filter
    LaplacianFilter = np.array([[1,1,1],[1,-8,1],[1,1,1]])
    LaplacianImage = filter2d(inputImage, LaplacianFilter)
    LaplacianImage = centralize(LaplacianImage)
    LaplacianImage = scalingComplex(LaplacianImage)
    LaplacianImage = gama_correction(LaplacianImage, 0.2)
    cv2.imwrite("LaplacianImageDFT.png", LaplacianImage)
